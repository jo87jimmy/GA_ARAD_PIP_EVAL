"""
在規則形狀合成測試集上評估學生模型的表面瑕疵檢測能力。
採用 MVTec-AD 官方推薦的三大黃金指標：
  1. 影像級 AUROC (Image-level AUROC)
  2. 像素級 AUROC (Pixel-level AUROC)
  3. PRO-score (Per-Region Overlap)

用法:
    python eval_regular_testset.py --obj_id 1 [--testset_root ./regular_testset] [--gpu_id -2]
"""
import os
import sys
import glob
import argparse
import numpy as np
import torch
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from model_unet import ReconstructiveSubNetwork, DiscriminativeSubNetwork
import random
from sklearn.metrics import roc_auc_score, roc_curve, auc
from scipy import ndimage


# =======================
# Utilities
# =======================
def setup_seed(seed):
    """固定所有隨機種子，確保實驗可重現性"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def compute_pro_score(gt_masks, pred_scores, num_thresholds=300, fpr_limit=0.3):
    """
    計算 PRO-score (Per-Region Overlap)。
    MVTec-AD 官方強推的指標：計算每一個獨立異常區塊（Region）被模型預測覆蓋的比例。

    演算法步驟：
      1. 對每張 GT mask 做 connected-component labeling，取出每個獨立異常區域
      2. 對每個閾值，計算每個區域被預測覆蓋的比例 (region overlap)
      3. 取所有區域的平均覆蓋率作為該閾值下的 PRO 值
      4. 以 FPR 為 x 軸、PRO 為 y 軸繪製曲線，取 FPR ≤ fpr_limit 範圍內的 AUC 並正規化

    Args:
        gt_masks: list of 2D numpy arrays (H, W)，值為 0 或 1
        pred_scores: list of 2D numpy arrays (H, W)，值為連續型異常分數
        num_thresholds: 閾值取樣數量
        fpr_limit: FPR 上限（MVTec 預設為 0.3）

    Returns:
        pro_auc: 正規化後的 PRO-score (0~1)
    """
    # 收集所有預測分數以決定閾值範圍
    all_scores = np.concatenate([s.ravel() for s in pred_scores])
    thresholds = np.linspace(all_scores.max(), all_scores.min(), num_thresholds)

    # 預先計算所有正常像素總數（用於 FPR 計算）
    all_gt = np.concatenate([g.ravel() for g in gt_masks])
    total_normal_pixels = np.sum(all_gt == 0)

    # 預先對每張 GT mask 做 connected-component labeling
    labeled_masks = []
    num_regions_per_image = []
    for gt in gt_masks:
        gt_binary = (gt > 0.5).astype(np.int32)
        labeled, num_features = ndimage.label(gt_binary)
        labeled_masks.append(labeled)
        num_regions_per_image.append(num_features)

    pro_values = []  # 每個閾值對應的平均 region overlap
    fpr_values = []  # 每個閾值對應的 FPR

    for threshold in thresholds:
        # 計算全域 FPR：在所有正常像素中，被錯誤預測為異常的比例
        fp_count = 0
        region_overlaps = []

        for i in range(len(gt_masks)):
            pred_binary = (pred_scores[i] >= threshold).astype(np.int32)
            gt_binary = (gt_masks[i] > 0.5).astype(np.int32)

            # 累加 False Positive 像素數
            fp_count += np.sum(pred_binary[gt_binary == 0])

            # 計算每個獨立區域的覆蓋率
            labeled = labeled_masks[i]
            num_regions = num_regions_per_image[i]
            for region_id in range(1, num_regions + 1):
                region_mask = (labeled == region_id)
                region_size = np.sum(region_mask)
                if region_size == 0:
                    continue
                # 該區域被預測覆蓋的比例
                overlap = np.sum(pred_binary[region_mask]) / region_size
                region_overlaps.append(overlap)

        # FPR = FP / Total Normal Pixels
        fpr = fp_count / max(total_normal_pixels, 1)
        # PRO = 所有區域覆蓋率的平均值
        pro = np.mean(region_overlaps) if region_overlaps else 0.0

        fpr_values.append(fpr)
        pro_values.append(pro)

    fpr_values = np.array(fpr_values)
    pro_values = np.array(pro_values)

    # 只取 FPR ≤ fpr_limit 的部分計算 AUC
    valid_idx = fpr_values <= fpr_limit
    if np.sum(valid_idx) < 2:
        return 0.0

    fpr_valid = fpr_values[valid_idx]
    pro_valid = pro_values[valid_idx]

    # 依 FPR 遞增排序
    sort_idx = np.argsort(fpr_valid)
    fpr_sorted = fpr_valid[sort_idx]
    pro_sorted = pro_valid[sort_idx]

    # 用 trapz 計算 AUC 並正規化到 [0, 1]
    pro_auc = np.trapezoid(pro_sorted, fpr_sorted) / fpr_limit
    return float(np.clip(pro_auc, 0.0, 1.0))


def get_available_gpu():
    if not torch.cuda.is_available():
        return -1
    gpu_count = torch.cuda.device_count()
    if gpu_count == 0:
        return -1
    gpu_memory = []
    for i in range(gpu_count):
        torch.cuda.set_device(i)
        gpu_memory.append((i, torch.cuda.memory_allocated(i)))
    return min(gpu_memory, key=lambda x: x[1])[0]


# =======================
# Dataset for regular shape testset
# =======================
class RegularShapeTestDataset(Dataset):
    """載入 regular_testset 中的圖片與對應 mask"""

    def __init__(self, root_dir, resize_shape=None):
        """
        root_dir: e.g. ./regular_testset/bottle/test/
        會掃描 root_dir 下所有子目錄 (good/, regular_shape/) 的 png
        """
        self.root_dir = root_dir
        self.resize_shape = resize_shape
        self.images = sorted(glob.glob(os.path.join(root_dir, '*', '*.png')))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        dir_path, file_name = os.path.split(img_path)
        base_dir = os.path.basename(dir_path)

        # 讀取圖片
        image = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if self.resize_shape is not None:
            image = cv2.resize(image, (self.resize_shape[1], self.resize_shape[0]))
        image = image.astype(np.float32) / 255.0

        if base_dir == 'good':
            mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.float32)
            has_anomaly = np.array([0], dtype=np.float32)
        else:
            # 找對應的 mask
            gt_dir = os.path.join(
                os.path.dirname(os.path.dirname(dir_path)),  # 上兩層
                'ground_truth', base_dir
            )
            mask_name = file_name.replace('.png', '_mask.png')
            mask_path = os.path.join(gt_dir, mask_name)
            if os.path.exists(mask_path):
                mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                if self.resize_shape is not None:
                    mask = cv2.resize(mask, (self.resize_shape[1], self.resize_shape[0]))
                mask = mask.astype(np.float32) / 255.0
            else:
                mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.float32)
            has_anomaly = np.array([1], dtype=np.float32)

        # Transpose to (C, H, W)
        image_t = np.transpose(image, (2, 0, 1))
        mask_t = mask.reshape((1, mask.shape[0], mask.shape[1]))

        return {
            'image': image_t,
            'has_anomaly': has_anomaly,
            'mask': mask_t,
            'idx': idx,
            'path': img_path,
            'category': base_dir,
        }


# =======================
# Main Evaluation
# =======================
def main(obj_names, args):
    setup_seed(111)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    save_root = "./inference_results_regular"
    os.makedirs(save_root, exist_ok=True)

    print(f"🔄 表面瑕疵檢測評估（三大黃金指標），共 {len(obj_names)} 個物件類別")

    # 收集所有類別的三大指標結果
    # 結構：{obj_name: {"image_auroc": float, "pixel_auroc": float, "pro_score": float} or None}
    metrics_summary = {}

    for obj_name in obj_names:
        img_dim = 256

        # --- 載入模型 ---
        student_model = ReconstructiveSubNetwork(in_channels=3, out_channels=3, base_width=64)
        recon_path = f'./student_model_checkpoints/{obj_name}_best_recon.pckl'
        if not os.path.exists(recon_path):
            print(f"❌ 未找到權重: {recon_path}")
            continue
        student_model.load_state_dict(torch.load(recon_path, map_location=device))
        student_model.to(device).eval()

        seg_model = DiscriminativeSubNetwork(in_channels=6, out_channels=2, base_channels=32)
        seg_path = f'./student_model_checkpoints/{obj_name}_best_seg.pckl'
        if not os.path.exists(seg_path):
            print(f"❌ 未找到權重: {seg_path}")
            continue
        seg_model.load_state_dict(torch.load(seg_path, map_location=device))
        seg_model.to(device).eval()

        # --- 載入測試集 ---
        test_dir = os.path.join(args.testset_root, obj_name, 'test')
        if not os.path.isdir(test_dir):
            print(f"⚠️ 跳過 {obj_name}: 找不到 {test_dir} (請先執行 generate_regular_testset.py)")
            continue

        dataset = RegularShapeTestDataset(test_dir, resize_shape=[img_dim, img_dim])
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
        print(f"\n📊 {obj_name}: {len(dataset)} 張圖片")

        output_dir = os.path.join(save_root, obj_name)
        os.makedirs(output_dir, exist_ok=True)

        # --- 收集結果（同時收集像素級與影像級資料，用於計算三大指標）---
        results = []            # (category, filename, image_score, gt_label) 用於逐圖日誌輸出
        img_labels = []         # 影像級 GT 標籤 (0=good, 1=anomaly)
        img_scores = []         # 影像級異常分數
        pixel_gt_list = []      # 像素級 GT mask (list of 2D arrays)
        pixel_pred_list = []    # 像素級預測異常分數圖 (list of 2D arrays)

        for i_batch, sample in enumerate(dataloader):
            gray_batch = sample['image'].to(device)
            has_anomaly = sample['has_anomaly'].numpy()[0, 0]
            true_mask = sample['mask']
            img_path = sample['path'][0]
            category = sample['category'][0]

            with torch.no_grad():
                gray_rec = student_model(gray_batch)
                joined_in = torch.cat((gray_rec, gray_batch), dim=1)
                out_mask = seg_model(joined_in)
                out_mask_sm = torch.softmax(out_mask, dim=1)

            # Image-level score：透過平均池化取最大值作為影像級異常分數
            out_mask_averaged = torch.nn.functional.avg_pool2d(
                out_mask_sm[:, 1:, :, :], 21, stride=1, padding=21 // 2
            ).cpu().numpy()
            image_score = float(np.max(out_mask_averaged))

            # 像素級預測熱圖與 GT mask（用於 Pixel AUROC 和 PRO-score）
            heatmap_np = out_mask_sm[0, 1, :, :].cpu().numpy()   # (H, W) 異常機率
            gt_mask_np = true_mask[0, 0, :, :].numpy()            # (H, W) 0/1 mask

            # 收集影像級資料
            img_labels.append(int(has_anomaly > 0.5))
            img_scores.append(image_score)

            # 收集像素級資料（每張圖完整保留，PRO-score 需要逐圖的 connected components）
            pixel_gt_list.append(gt_mask_np)
            pixel_pred_list.append(heatmap_np)

            # 取得顯示用的數據
            original_np = gray_batch.permute(0, 2, 3, 1).cpu().numpy()[0]
            original_np = (original_np - original_np.min()) / (original_np.max() - original_np.min() + 1e-8)
            # BGR -> RGB
            original_np = original_np[:, :, ::-1].copy()

            fname = os.path.basename(img_path)
            gt_label = "anomaly" if has_anomaly > 0.5 else "good"
            results.append((category, fname, image_score, gt_label))

            # --- 繪製三欄圖: 原圖 | 預測熱圖 | GT mask ---
            fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

            axes[0].imshow(original_np)
            axes[0].set_title(f'Original [{category}]', fontsize=10)
            axes[0].axis('off')

            im = axes[1].imshow(heatmap_np, cmap='hot', vmin=0, vmax=1)
            axes[1].set_title(f'Anomaly Heatmap\nScore: {image_score:.4f}', fontsize=10)
            axes[1].axis('off')
            plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

            axes[2].imshow(gt_mask_np, cmap='gray', vmin=0, vmax=1)
            axes[2].set_title(f'GT Mask [{gt_label}]', fontsize=10)
            axes[2].axis('off')

            plt.suptitle(f'{obj_name} — {fname}', fontsize=12, fontweight='bold')
            plt.tight_layout()

            save_path = os.path.join(output_dir, f'{category}_{fname}')
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.show()
            plt.close()

        # ================================================================
        # 按難度等級分組計算三大黃金指標
        # （支援舊格式「單一 regular_shape」與新格式「多難度等級」）
        # ================================================================

        # 辨識難度等級：所有非 'good' 的 category 子目錄
        difficulty_levels = sorted(set(
            cat for cat, _, _, _ in results if cat != 'good'
        ))
        # Good 圖片的索引（每個難度等級都會共用這些作為負樣本）
        good_indices = [i for i, (cat, _, _, _) in enumerate(results) if cat == 'good']

        # 用於儲存該物件的per-difficulty指標
        per_diff_metrics = {}

        def _compute_subset_metrics(indices):
            """計算指定索引子集的三大指標（共用工具函式）"""
            s_labels = [img_labels[i] for i in indices]
            s_scores = [img_scores[i] for i in indices]
            s_pgt = [pixel_gt_list[i] for i in indices]
            s_ppred = [pixel_pred_list[i] for i in indices]
            m = {"image_auroc": None, "pixel_auroc": None, "pro_score": None}
            # Image AUROC
            if len(set(s_labels)) >= 2:
                m["image_auroc"] = roc_auc_score(s_labels, s_scores)
            # Pixel AUROC
            a_gt = np.concatenate([g.ravel() for g in s_pgt])
            a_pred = np.concatenate([p.ravel() for p in s_ppred])
            if len(set(a_gt.astype(int))) >= 2:
                m["pixel_auroc"] = roc_auc_score(a_gt, a_pred)
            # PRO-score
            anom_gt = [g for g, lbl in zip(s_pgt, s_labels) if lbl == 1]
            has_r = any(np.sum(g > 0.5) > 0 for g in anom_gt) if anom_gt else False
            if has_r:
                m["pro_score"] = compute_pro_score(s_pgt, s_ppred)
            return m

        # --- 按難度分組計算 ---
        print(f"\n{'='*90}")
        print(f"  {obj_name} — 泛化能力評估（按難度等級）")
        print(f"{'='*90}")
        print(f"{'Difficulty':<25} {'#Anom':>6} {'#Good':>6}   {'Image AUROC':>12} {'Pixel AUROC':>12} {'PRO-score':>12}")
        print(f"{'-'*90}")

        for diff_name in difficulty_levels:
            diff_indices = [i for i, (cat, _, _, _) in enumerate(results) if cat == diff_name]
            eval_indices = good_indices + diff_indices
            m = _compute_subset_metrics(eval_indices)
            per_diff_metrics[diff_name] = m

            i_str = f"{m['image_auroc']:.4f}" if m['image_auroc'] is not None else "N/A"
            p_str = f"{m['pixel_auroc']:.4f}" if m['pixel_auroc'] is not None else "N/A"
            r_str = f"{m['pro_score']:.4f}" if m['pro_score'] is not None else "N/A"
            print(f"{diff_name:<25} {len(diff_indices):>6} {len(good_indices):>6}   {i_str:>12} {p_str:>12} {r_str:>12}")

        # --- Overall（全難度彙總）---
        all_indices = list(range(len(results)))
        overall_m = _compute_subset_metrics(all_indices)
        n_anom = sum(1 for cat, _, _, _ in results if cat != 'good')
        i_str = f"{overall_m['image_auroc']:.4f}" if overall_m['image_auroc'] is not None else "N/A"
        p_str = f"{overall_m['pixel_auroc']:.4f}" if overall_m['pixel_auroc'] is not None else "N/A"
        r_str = f"{overall_m['pro_score']:.4f}" if overall_m['pro_score'] is not None else "N/A"
        print(f"{'-'*90}")
        print(f"{'Overall (all levels)':<25} {n_anom:>6} {len(good_indices):>6}   {i_str:>12} {p_str:>12} {r_str:>12}")
        print(f"{'='*90}")

        metrics_summary[obj_name] = {
            "per_difficulty": per_diff_metrics,
            "overall": overall_m,
        }

    # =======================
    # 所有類別 — 按難度等級的泛化能力總表
    # =======================
    if metrics_summary:
        # 收集所有出現過的難度等級（跨類別取聯集）
        all_diff_levels = sorted(set(
            d for v in metrics_summary.values() for d in v["per_difficulty"]
        ))

        # --- 1. 每個難度等級的跨類別平均 ---
        print(f"\n{'='*90}")
        print(f"  所有類別 — 泛化能力總表（按難度等級，跨類別平均）")
        print(f"{'='*90}")
        print(f"{'Difficulty Level':<25} {'Image AUROC':>14} {'Pixel AUROC':>14} {'PRO-score':>14}")
        print(f"{'-'*90}")

        # 用於繪製泛化曲線的資料
        curve_data = {"level": [], "image_auroc": [], "pixel_auroc": [], "pro_score": []}

        for diff in all_diff_levels:
            vals_i, vals_p, vals_r = [], [], []
            for obj_data in metrics_summary.values():
                m = obj_data["per_difficulty"].get(diff)
                if m:
                    if m["image_auroc"] is not None: vals_i.append(m["image_auroc"])
                    if m["pixel_auroc"] is not None: vals_p.append(m["pixel_auroc"])
                    if m["pro_score"] is not None: vals_r.append(m["pro_score"])

            mi = np.mean(vals_i) if vals_i else None
            mp = np.mean(vals_p) if vals_p else None
            mr = np.mean(vals_r) if vals_r else None
            mi_s = f"{mi:.4f}" if mi is not None else "N/A"
            mp_s = f"{mp:.4f}" if mp is not None else "N/A"
            mr_s = f"{mr:.4f}" if mr is not None else "N/A"
            print(f"{diff:<25} {mi_s:>14} {mp_s:>14} {mr_s:>14}")

            curve_data["level"].append(diff)
            curve_data["image_auroc"].append(mi)
            curve_data["pixel_auroc"].append(mp)
            curve_data["pro_score"].append(mr)

        # Overall（全難度全類別平均）
        ov_i, ov_p, ov_r = [], [], []
        for obj_data in metrics_summary.values():
            m = obj_data["overall"]
            if m["image_auroc"] is not None: ov_i.append(m["image_auroc"])
            if m["pixel_auroc"] is not None: ov_p.append(m["pixel_auroc"])
            if m["pro_score"] is not None: ov_r.append(m["pro_score"])
        print(f"{'-'*90}")
        ov_i_s = f"{np.mean(ov_i):.4f}" if ov_i else "N/A"
        ov_p_s = f"{np.mean(ov_p):.4f}" if ov_p else "N/A"
        ov_r_s = f"{np.mean(ov_r):.4f}" if ov_r else "N/A"
        print(f"{'Overall Mean':<25} {ov_i_s:>14} {ov_p_s:>14} {ov_r_s:>14}")
        print(f"{'='*90}")

        # --- 2. 繪製泛化曲線圖（指標 vs 難度等級）---
        valid_levels = [i for i, l in enumerate(curve_data["level"])
                        if curve_data["image_auroc"][i] is not None]
        if len(valid_levels) >= 2:
            x_labels = [curve_data["level"][i].replace("level_", "L") for i in valid_levels]
            x_pos = list(range(len(valid_levels)))

            fig, ax = plt.subplots(figsize=(10, 6))
            for metric_key, label, color, marker in [
                ("image_auroc", "Image AUROC", "#E74C3C", "o"),
                ("pixel_auroc", "Pixel AUROC", "#3498DB", "s"),
                ("pro_score",   "PRO-score",   "#2ECC71", "^"),
            ]:
                y = [curve_data[metric_key][i] for i in valid_levels]
                if any(v is not None for v in y):
                    y_clean = [v if v is not None else 0 for v in y]
                    ax.plot(x_pos, y_clean, marker=marker, label=label,
                            color=color, linewidth=2.5, markersize=8)
                    for xi, yi in zip(x_pos, y_clean):
                        ax.annotate(f"{yi:.3f}", (xi, yi), textcoords="offset points",
                                    xytext=(0, 10), ha='center', fontsize=8)

            ax.set_xticks(x_pos)
            ax.set_xticklabels(x_labels, fontsize=9)
            ax.set_xlabel("Difficulty Level", fontsize=12)
            ax.set_ylabel("Metric Value", fontsize=12)
            ax.set_title("Generalization Curve — Metrics vs Difficulty", fontsize=14, fontweight='bold')
            ax.set_ylim(-0.05, 1.1)
            ax.legend(fontsize=11, loc='lower left')
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            curve_path = os.path.join(save_root, "generalization_curve.png")
            plt.savefig(curve_path, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"\n📈 泛化曲線圖已儲存: {curve_path}")

        # --- 3. 繪製總表圖片 ---
        table_data = []
        for diff in all_diff_levels:
            cd = curve_data
            idx = cd["level"].index(diff)
            table_data.append([
                diff,
                f"{cd['image_auroc'][idx]:.4f}" if cd['image_auroc'][idx] is not None else "N/A",
                f"{cd['pixel_auroc'][idx]:.4f}" if cd['pixel_auroc'][idx] is not None else "N/A",
                f"{cd['pro_score'][idx]:.4f}" if cd['pro_score'][idx] is not None else "N/A",
            ])
        table_data.append(["Overall Mean", ov_i_s, ov_p_s, ov_r_s])

        col_labels = ["Difficulty", "Image AUROC", "Pixel AUROC", "PRO-score"]
        n_rows = len(table_data)
        fig_height = max(2.5, 0.45 * n_rows + 1.2)
        fig, ax = plt.subplots(figsize=(10, fig_height))
        ax.axis('off')
        ax.set_title("Generalization Summary — Golden Three Metrics by Difficulty",
                     fontsize=13, fontweight='bold', pad=12)
        table = ax.table(cellText=table_data, colLabels=col_labels,
                         loc='center', cellLoc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.0, 1.4)
        for j in range(len(col_labels)):
            table[0, j].set_facecolor('#4472C4')
            table[0, j].set_text_props(color='white', fontweight='bold')
        n_diff = len(all_diff_levels)
        for i in range(1, n_rows + 1):
            for j in range(len(col_labels)):
                if i > n_diff:
                    table[i, j].set_facecolor('#D9E2F3')
                    table[i, j].set_text_props(fontweight='bold')
                elif i % 2 == 0:
                    table[i, j].set_facecolor('#F2F2F2')
        plt.tight_layout()
        table_path = os.path.join(save_root, "summary_table_by_difficulty.png")
        plt.savefig(table_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"📊 難度分級總表已儲存: {table_path}")

    print("\n🎉 表面瑕疵檢測評估完成！")


# =======================
# Entry
# =======================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='規則形狀合成測試集 — 泛化能力評估')
    parser.add_argument('--obj_id', type=int, required=True,
                        help='物件 ID (0-14)，-1 表示全部')
    parser.add_argument('--gpu_id', type=int, default=-2,
                        help='GPU ID (-2: auto, -1: CPU)')
    parser.add_argument('--testset_root', type=str, default='./regular_testset',
                        help='規則形狀測試集根目錄')
    parser.add_argument('--mvtec_root', type=str, default='./mvtec',
                        help='MVTec 資料集根目錄 (未使用，保持介面一致)')
    args = parser.parse_args()

    obj_list = [
        'capsule', 'bottle', 'carpet', 'leather', 'pill', 'transistor',
        'tile', 'cable', 'zipper', 'toothbrush', 'metal_nut', 'hazelnut',
        'screw', 'grid', 'wood'
    ]

    if args.gpu_id == -2:
        args.gpu_id = get_available_gpu()
        print(f"自動選擇 GPU: {args.gpu_id}")

    if args.obj_id == -1:
        picked = obj_list
    else:
        picked = [obj_list[args.obj_id]]

    if args.gpu_id == -1:
        main(picked, args)
    else:
        with torch.cuda.device(args.gpu_id):
            main(picked, args)
