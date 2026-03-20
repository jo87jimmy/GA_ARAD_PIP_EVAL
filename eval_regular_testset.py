"""
在規則形狀合成測試集上評估學生模型的泛化能力。
顯示每張圖的原圖 + 異常熱圖 + GT mask，並輸出 Image-level 異常分數。

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
from sklearn.metrics import roc_auc_score


# =======================
# Utilities
# =======================
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


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

    print(f"🔄 規則形狀泛化測試，共 {len(obj_names)} 個物件類別")

    # 收集所有類別的 AUROC 結果
    auroc_summary = {}  # {obj_name: auroc or None}

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

        # --- 收集結果 ---
        results = []  # (category, filename, image_score, gt_label)

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

            # Image-level score
            out_mask_averaged = torch.nn.functional.avg_pool2d(
                out_mask_sm[:, 1:, :, :], 21, stride=1, padding=21 // 2
            ).cpu().numpy()
            image_score = float(np.max(out_mask_averaged))

            # 取得顯示用的數據
            original_np = gray_batch.permute(0, 2, 3, 1).cpu().numpy()[0]
            original_np = (original_np - original_np.min()) / (original_np.max() - original_np.min() + 1e-8)
            # BGR -> RGB
            original_np = original_np[:, :, ::-1].copy()

            heatmap_np = out_mask_sm[0, 1, :, :].cpu().numpy()
            gt_mask_np = true_mask[0, 0, :, :].numpy()

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

        # --- Image-level 統計表 ---
        print(f"\n{'='*70}")
        print(f"  {obj_name} — Image-level 泛化測試結果")
        print(f"{'='*70}")
        print(f"{'Category':<20} {'Filename':<40} {'Score':>8} {'GT':>8}")
        print(f"{'-'*70}")

        scores_anomaly = []
        scores_good = []

        for cat, fname, score, gt in results:
            tag = "✅" if (gt == "good" and score < 0.5) or (gt == "anomaly" and score >= 0.5) else "❌"
            print(f"{cat:<20} {fname:<40} {score:>8.4f} {gt:>8} {tag}")
            if gt == "anomaly":
                scores_anomaly.append(score)
            else:
                scores_good.append(score)

        print(f"{'-'*70}")
        if scores_anomaly:
            print(f"  異常圖 (regular_shape) 平均分數: {np.mean(scores_anomaly):.4f}  "
                  f"(min={np.min(scores_anomaly):.4f}, max={np.max(scores_anomaly):.4f})")
        if scores_good:
            print(f"  正常圖 (good)          平均分數: {np.mean(scores_good):.4f}  "
                  f"(min={np.min(scores_good):.4f}, max={np.max(scores_good):.4f})")

        # 簡單計算 detection rate (threshold=0.5)
        if scores_anomaly:
            detected = sum(1 for s in scores_anomaly if s >= 0.5)
            print(f"  異常偵測率 (threshold=0.5): {detected}/{len(scores_anomaly)} "
                  f"= {detected/len(scores_anomaly)*100:.1f}%")
        if scores_good:
            fp = sum(1 for s in scores_good if s >= 0.5)
            print(f"  正常誤報率 (threshold=0.5): {fp}/{len(scores_good)} "
                  f"= {fp/len(scores_good)*100:.1f}%")

        # 計算 Image-level AUROC
        gt_labels = [1 if gt == "anomaly" else 0 for _, _, _, gt in results]
        pred_scores = [score for _, _, score, _ in results]
        if len(set(gt_labels)) >= 2:
            auroc = roc_auc_score(gt_labels, pred_scores)
            print(f"  Image AUROC: {auroc:.4f}")
            auroc_summary[obj_name] = auroc
        else:
            print(f"  Image AUROC: N/A (只有單一類別的樣本)")
            auroc_summary[obj_name] = None
        print(f"{'='*70}\n")

    # =======================
    # 所有類別的 Image AUROC 泛化總表
    # =======================
    print(f"\n{'='*50}")
    print(f"  Image AUROC 泛化總表 (Regular Shape Testset)")
    print(f"{'='*50}")
    print(f"{'Category':<20} {'Image AUROC':>12}")
    print(f"{'-'*50}")

    valid_aurocs = []
    for obj_name in obj_names:
        if obj_name in auroc_summary:
            auroc = auroc_summary[obj_name]
            if auroc is not None:
                print(f"{obj_name:<20} {auroc:>12.4f}")
                valid_aurocs.append(auroc)
            else:
                print(f"{obj_name:<20} {'N/A':>12}")
        else:
            print(f"{obj_name:<20} {'SKIP':>12}")

    print(f"{'-'*50}")
    if valid_aurocs:
        print(f"  Image AUROC (regular_shape) 平均分數: {np.mean(valid_aurocs):.4f}  "
              f"(min={np.min(valid_aurocs):.4f}, max={np.max(valid_aurocs):.4f})")
    print(f"{'='*50}")

    print("\n🎉 規則形狀泛化測試完成！")


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
