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
    pro_auc = np.trapz(pro_sorted, fpr_sorted) / fpr_limit
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
        # 計算該類別的三大黃金指標
        # ================================================================
        category_metrics = {"image_auroc": None, "pixel_auroc": None, "pro_score": None}

        # --- 1. 影像級 AUROC ---
        # 需要同時有正常和異常樣本才能計算 AUROC
        if len(set(img_labels)) >= 2:
            category_metrics["image_auroc"] = roc_auc_score(img_labels, img_scores)
            print(f"  📈 Image-level AUROC: {category_metrics['image_auroc']:.4f}")
        else:
            print(f"  ⚠️ Image-level AUROC: N/A（僅有單一類別樣本，無法計算）")

        # --- 2. 像素級 AUROC ---
        # 將所有像素展平後計算 AUROC
        all_pixel_gt = np.concatenate([g.ravel() for g in pixel_gt_list])
        all_pixel_pred = np.concatenate([p.ravel() for p in pixel_pred_list])
        if len(set(all_pixel_gt.astype(int))) >= 2:
            category_metrics["pixel_auroc"] = roc_auc_score(all_pixel_gt, all_pixel_pred)
            print(f"  📈 Pixel-level AUROC: {category_metrics['pixel_auroc']:.4f}")
        else:
            print(f"  ⚠️ Pixel-level AUROC: N/A（像素級僅有單一類別，無法計算）")

        # --- 3. PRO-score (Per-Region Overlap) ---
        # 僅對含有異常區域的圖片計算
        anomaly_gt_masks = [g for g, lbl in zip(pixel_gt_list, img_labels) if lbl == 1]
        anomaly_pred_scores = [p for p, lbl in zip(pixel_pred_list, img_labels) if lbl == 1]
        # 確認至少有一張異常圖且其中存在異常區域
        has_anomaly_regions = any(np.sum(g > 0.5) > 0 for g in anomaly_gt_masks) if anomaly_gt_masks else False
        if has_anomaly_regions:
            category_metrics["pro_score"] = compute_pro_score(pixel_gt_list, pixel_pred_list)
            print(f"  📈 PRO-score:         {category_metrics['pro_score']:.4f}")
        else:
            print(f"  ⚠️ PRO-score: N/A（無異常區域可計算）")

        # --- 逐圖日誌輸出 ---
        print(f"\n{'='*70}")
        print(f"  {obj_name} — 逐圖推論結果")
        print(f"{'='*70}")
        print(f"{'Category':<20} {'Filename':<40} {'Score':>8} {'GT':>8}")
        print(f"{'-'*70}")
        for cat, fname, score, gt in results:
            tag = "✅" if (gt == "good" and score < 0.5) or (gt == "anomaly" and score >= 0.5) else "❌"
            print(f"{cat:<20} {fname:<40} {score:>8.4f} {gt:>8} {tag}")
        print(f"{'-'*70}")

        metrics_summary[obj_name] = category_metrics

    # =======================
    # 所有類別的三大黃金指標總表
    # =======================
    if metrics_summary:
        # --- 終端輸出 ---
        print(f"\n{'='*80}")
        print(f"  所有類別 — 表面瑕疵檢測黃金三大指標總表")
        print(f"{'='*80}")
        print(f"{'Category':<20} {'Image AUROC':>14} {'Pixel AUROC':>14} {'PRO-score':>14}")
        print(f"{'-'*80}")

        # 收集有效數值以計算 Overall Mean
        valid_image_aurocs = []
        valid_pixel_aurocs = []
        valid_pro_scores = []

        for obj_name, metrics in metrics_summary.items():
            img_str = f"{metrics['image_auroc']:.4f}" if metrics['image_auroc'] is not None else "N/A"
            pix_str = f"{metrics['pixel_auroc']:.4f}" if metrics['pixel_auroc'] is not None else "N/A"
            pro_str = f"{metrics['pro_score']:.4f}" if metrics['pro_score'] is not None else "N/A"
            print(f"{obj_name:<20} {img_str:>14} {pix_str:>14} {pro_str:>14}")

            if metrics['image_auroc'] is not None:
                valid_image_aurocs.append(metrics['image_auroc'])
            if metrics['pixel_auroc'] is not None:
                valid_pixel_aurocs.append(metrics['pixel_auroc'])
            if metrics['pro_score'] is not None:
                valid_pro_scores.append(metrics['pro_score'])

        print(f"{'-'*80}")
        # 計算各指標的 Overall Mean
        mean_img = f"{np.mean(valid_image_aurocs):.4f}" if valid_image_aurocs else "N/A"
        mean_pix = f"{np.mean(valid_pixel_aurocs):.4f}" if valid_pixel_aurocs else "N/A"
        mean_pro = f"{np.mean(valid_pro_scores):.4f}" if valid_pro_scores else "N/A"
        print(f"{'Overall Mean':<20} {mean_img:>14} {mean_pix:>14} {mean_pro:>14}")
        print(f"{'='*80}")

        # --- 繪製三大指標表格圖片 ---
        table_data = []
        for obj_name, metrics in metrics_summary.items():
            table_data.append([
                obj_name,
                f"{metrics['image_auroc']:.4f}" if metrics['image_auroc'] is not None else "N/A",
                f"{metrics['pixel_auroc']:.4f}" if metrics['pixel_auroc'] is not None else "N/A",
                f"{metrics['pro_score']:.4f}" if metrics['pro_score'] is not None else "N/A",
            ])
        # 附加 Overall Mean 列
        table_data.append(["Overall Mean", mean_img, mean_pix, mean_pro])

        col_labels = ["Category", "Image AUROC", "Pixel AUROC", "PRO-score"]
        n_rows = len(table_data)
        fig_height = max(2.5, 0.45 * n_rows + 1.2)
        fig, ax = plt.subplots(figsize=(10, fig_height))
        ax.axis('off')
        ax.set_title("Surface Defect Detection — Golden Three Metrics Summary",
                     fontsize=13, fontweight='bold', pad=12)

        table = ax.table(cellText=table_data, colLabels=col_labels,
                         loc='center', cellLoc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.0, 1.4)

        # 標頭樣式（深藍底白字）
        for j in range(len(col_labels)):
            table[0, j].set_facecolor('#4472C4')
            table[0, j].set_text_props(color='white', fontweight='bold')

        # 資料列交替顏色 + Overall Mean 列特殊高亮
        n_categories = len(metrics_summary)
        for i in range(1, n_rows + 1):
            is_summary_row = (i > n_categories)
            for j in range(len(col_labels)):
                if is_summary_row:
                    table[i, j].set_facecolor('#D9E2F3')
                    table[i, j].set_text_props(fontweight='bold')
                elif i % 2 == 0:
                    table[i, j].set_facecolor('#F2F2F2')

        plt.tight_layout()
        table_path = os.path.join(save_root, "summary_table_golden_metrics.png")
        plt.savefig(table_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"\n📊 黃金三大指標總表已儲存: {table_path}")

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
