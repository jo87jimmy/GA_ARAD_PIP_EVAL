import os
import torch
from torchvision import transforms as T
import numpy as np
import random  # 亂數控制
import argparse  # 命令列參數處理
from data_loader import MVTecDRAEM_Test_Visual_Dataset
from model_unet import ReconstructiveSubNetwork, DiscriminativeSubNetwork
import torchvision.transforms as transforms
import cv2
from PIL import Image  # 雖然 transform 用到了，但直接用 cv2 讀寫更一致
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt


def setup_seed(seed):
    # 設定隨機種子，確保實驗可重現
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True  # 保證結果可重現
    torch.backends.cudnn.benchmark = False  # 關閉自動最佳化搜尋


# =======================
# Utilities
# =======================
def get_available_gpu():
    """自動選擇記憶體使用率最低的GPU"""
    if not torch.cuda.is_available():
        return -1  # 沒有GPU可用

    gpu_count = torch.cuda.device_count()
    if gpu_count == 0:
        return -1

    # 檢查每個GPU的記憶體使用情況
    gpu_memory = []
    for i in range(gpu_count):
        torch.cuda.set_device(i)
        memory_allocated = torch.cuda.memory_allocated(i)
        # memory_reserved = torch.cuda.memory_reserved(i) # 這個在某些情況下會顯示較高，我們更關注已分配的
        gpu_memory.append((i, memory_allocated))  # 只用 allocated

    # 選擇記憶體使用最少的GPU
    available_gpu = min(gpu_memory, key=lambda x: x[1])[0]
    return available_gpu


# =======================
# Main Pipeline
# =======================
def main(obj_names, args):
    setup_seed(111)  # 固定隨機種子
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # 建立主存檔資料夾
    save_root = "./inference_results"  # 推理結果通常保存在不同的目錄
    if not os.path.exists(save_root):
        os.makedirs(save_root)
    print("🔄 開始測試，共有物件類別:", len(obj_names))
    for obj_name in obj_names:
        img_dim = 256
        student_model = ReconstructiveSubNetwork(in_channels=3,
                                                 out_channels=3,
                                                 base_width=64)
        model_best_recon_weights_path = './student_model_checkpoints/' + obj_name + '_best_recon.pckl'  # ⬅️ 我的的權重路徑
        if not os.path.exists(model_best_recon_weights_path):
            print(
                f"❌ 錯誤: 未找到模型權重檔案: {model_best_recon_weights_path}，請檢查路徑或訓練是否完成。"
            )
            continue

        student_model.load_state_dict(
            torch.load(model_best_recon_weights_path, map_location=device))
        student_model.cuda()
        student_model.eval()

        student_seg_model = DiscriminativeSubNetwork(in_channels=6,
                                                     out_channels=2,
                                                     base_channels=32)
        model_best_seg_weights_path = './student_model_checkpoints/' + obj_name + '_best_seg.pckl'  # ⬅️ 我的的權重路徑
        if not os.path.exists(model_best_seg_weights_path):
            print(
                f"❌ 錯誤: 未找到模型權重檔案: {model_best_seg_weights_path}，請檢查路徑或訓練是否完成。"
            )
            continue

        student_seg_model.load_state_dict(
            torch.load(model_best_seg_weights_path, map_location=device))
        student_seg_model.cuda()
        student_seg_model.eval()

        # 建立資料集和資料載入器
        try:
            path = args.mvtec_root + "/" + obj_name + "/test/"
            print(f"📂 載入資料集路徑:{path}")

            # 检查test目录下的子目录
            subdirs = ['broken_large', 'broken_small', 'contamination', 'good']
            existing_subdirs = []

            for subdir in subdirs:
                subdir_path = os.path.join(path, subdir)
                if os.path.exists(subdir_path) and os.path.isdir(subdir_path):
                    existing_subdirs.append(subdir)
                    print(f"✅ 找到類別: {subdir}")

            if not existing_subdirs:
                raise Exception(f"在 {path} 中未找到任何測試類別目錄")

            dataset = MVTecDRAEM_Test_Visual_Dataset(
                path, resize_shape=[img_dim, img_dim])

            dataloader = DataLoader(dataset,
                                    batch_size=1,
                                    shuffle=False,
                                    num_workers=0)

        except Exception as e:
            print(f"❌ 錯誤: {e}")

        print(f"📊 資料集大小: {len(dataset)} 張圖片")

        total_pixel_scores = np.zeros((img_dim * img_dim * len(dataset)))
        total_gt_pixel_scores = np.zeros((img_dim * img_dim * len(dataset)))
        mask_cnt = 0

        anomaly_score_gt = []
        anomaly_score_prediction = []

        display_images = torch.zeros((16, 3, 256, 256)).cuda()
        display_gt_images = torch.zeros((16, 3, 256, 256)).cuda()
        display_out_masks = torch.zeros((16, 1, 256, 256)).cuda()
        display_in_masks = torch.zeros((16, 1, 256, 256)).cuda()
        cnt_display = 0
        display_indices = np.random.randint(len(dataloader), size=(16, ))

        for i_batch, sample_batched in enumerate(dataloader):
            # 建立該類別的輸出資料夾
            output_dir = os.path.join(save_root, obj_name)
            print(f"📂 載輸出資料夾路徑:{output_dir}")
            os.makedirs(output_dir, exist_ok=True)  # 確保資料夾存在

            gray_batch = sample_batched["image"].cuda()

            # 獲取原始圖像（用於顯示）
            original_image = gray_batch.permute(0, 2, 3, 1).cpu().numpy()[0]
            # 正規化到 [0, 1] 範圍
            original_image = (original_image - original_image.min()) / (
                original_image.max() - original_image.min())

            is_normal = sample_batched["has_anomaly"].detach().numpy()[0, 0]
            anomaly_score_gt.append(is_normal)
            true_mask = sample_batched["mask"]
            true_mask_cv = true_mask.detach().numpy()[0, :, :, :].transpose(
                (1, 2, 0))

            gray_rec = student_model(gray_batch)
            joined_in = torch.cat((gray_rec.detach(), gray_batch), dim=1)

            out_mask = student_seg_model(joined_in)
            out_mask_sm = torch.softmax(out_mask, dim=1)

            if i_batch in display_indices:
                t_mask = out_mask_sm[:, 1:, :, :]
                display_images[cnt_display] = gray_rec[0].cpu().detach()
                display_gt_images[cnt_display] = gray_batch[0].cpu().detach()
                display_out_masks[cnt_display] = t_mask[0].cpu().detach()
                display_in_masks[cnt_display] = true_mask[0].cpu().detach()
                cnt_display += 1

            out_mask_cv = out_mask_sm[0, 1, :, :].detach().cpu().numpy()
            save_path_base = os.path.join(output_dir, obj_name)

            # 在同一個plt中顯示原圖和異常熱圖
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

            # 顯示原圖
            ax1.imshow(original_image)
            ax1.set_title('Original Image')
            ax1.axis('off')

            # 顯示異常熱圖
            im = ax2.imshow(out_mask_cv, cmap='hot')
            ax2.set_title('Predicted Anomaly Heatmap')
            ax2.axis('off')

            # 添加顏色條
            plt.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)

            plt.tight_layout()

            # 存檔
            save_path_combined = f"{save_path_base}_combined_{str(i_batch)}.png"
            plt.savefig(save_path_combined,
                        dpi=300,
                        bbox_inches='tight',
                        pad_inches=0.1)
            print(f"Combined image saved to: {save_path_combined}")

            plt.show()
            plt.close()  # 關閉圖形以釋放記憶體

            out_mask_averaged = torch.nn.functional.avg_pool2d(
                out_mask_sm[:, 1:, :, :], 21, stride=1,
                padding=21 // 2).cpu().detach().numpy()
            image_score = np.max(out_mask_averaged)

            anomaly_score_prediction.append(image_score)

            flat_true_mask = true_mask_cv.flatten()
            flat_out_mask = out_mask_cv.flatten()
            total_pixel_scores[mask_cnt * img_dim * img_dim:(mask_cnt + 1) *
                               img_dim * img_dim] = flat_out_mask
            total_gt_pixel_scores[mask_cnt * img_dim * img_dim:(mask_cnt + 1) *
                                  img_dim * img_dim] = flat_true_mask
            mask_cnt += 1
        print(f"\n✅ 物件類別 {obj_name} 測試完成！")
    print("\n🎉 所有測試已完成！")


# =======================
# Run pipeline
# =======================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--obj_id', action='store', type=int, required=True)
    parser.add_argument('--gpu_id',
                        action='store',
                        type=int,
                        default=-2,
                        required=False,
                        help='GPU ID (-2: auto-select, -1: CPU)')
    parser.add_argument('--mvtec_root',
                        type=str,
                        default='./mvtec',
                        help='Path to the MVTec dataset root directory')
    parser.add_argument('--checkpoint_dir',
                        type=str,
                        default='./save_files',
                        help='Directory to load model checkpoints')

    args = parser.parse_args()

    # 自動選擇GPU
    if args.gpu_id == -2:  # 自動選擇模式
        args.gpu_id = get_available_gpu()
        print(f"自動選擇 GPU: {args.gpu_id}")

    obj_batch = [['capsule'], ['bottle'], ['carpet'], ['leather'], ['pill'],
                 ['transistor'], ['tile'], ['cable'], ['zipper'],
                 ['toothbrush'], ['metal_nut'], ['hazelnut'], ['screw'],
                 ['grid'], ['wood']]

    if int(args.obj_id) == -1:
        obj_list = [
            'capsule', 'bottle', 'carpet', 'leather', 'pill', 'transistor',
            'tile', 'cable', 'zipper', 'toothbrush', 'metal_nut', 'hazelnut',
            'screw', 'grid', 'wood'
        ]
        picked_classes = obj_list
    else:
        picked_classes = obj_batch[int(args.obj_id)]

    # 根據選擇的GPU執行
    if args.gpu_id == -1:
        # 使用CPU
        main(picked_classes, args)
    else:
        # 使用GPU
        with torch.cuda.device(args.gpu_id):
            main(picked_classes, args)
