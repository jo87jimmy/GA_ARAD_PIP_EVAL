import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from model_unet import AnomalyDetectionModel

# --- 載入模型與權重 ---

# 設定參數 (必須與訓練時學生模型的參數一致)
IMG_CHANNELS = 3
SEG_CLASSES = 2
STUDENT_RECON_BASE = 64
STUDENT_DISC_BASE = 64
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 實例化學生模型架構
student_model = AnomalyDetectionModel(
    recon_in=IMG_CHANNELS,
    recon_out=IMG_CHANNELS,
    recon_base=STUDENT_RECON_BASE,
    disc_in=IMG_CHANNELS * 2,
    disc_out=SEG_CLASSES,
    disc_base=STUDENT_DISC_BASE
).to(DEVICE)

# 載入訓練好的學生模型權重
model_weights_path = './student_model_checkpoints/student_model.pckl' # ⬅️ 修改為您的權重路徑
student_model.load_state_dict(torch.load(model_weights_path, map_location=DEVICE))

# --- 2. 設定為評估模式 ---
student_model.eval()

# --- 3. 定義一個完整的推論函數 ---

def predict_anomaly(model, image_path, device):
    """
    對單張圖片進行異常檢測推論

    Args:
        model (nn.Module): 訓練好的學生模型
        image_path (str): 輸入圖片的路徑
        device (str): 'cuda' or 'cpu'

    Returns:
        tuple: (原始圖像, 重建圖像, 異常遮罩) 均為 numpy array
    """
    # 定義圖像預處理流程 (應與訓練時的驗證集/測試集流程一致)
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)), # 假設模型輸入尺寸為 224x224
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # 載入並預處理圖像
    image = Image.open(image_path).convert("RGB")
    image_tensor = preprocess(image).unsqueeze(0).to(device) # 增加 batch 維度 [C, H, W] -> [1, C, H, W]

    # --- 4. 執行前向傳播 (在 no_grad 上下文中以節省資源) ---
    with torch.no_grad():
        # 推論時，我們只需要分割圖，但模型會同時返回重建圖
        # 我們不需要特徵圖，所以 return_feats=False
        recon_image_tensor, seg_map_logits = model(image_tensor, return_feats=False)

    # --- 5. 後處理輸出 ---

    # a. 處理分割圖
    # seg_map_logits 的形狀是 [1, 2, H, W]，其中 2 是類別數 (0:正常, 1:異常)
    # 使用 softmax 將 logits 轉換為機率
    seg_map_probs = torch.softmax(seg_map_logits, dim=1)
    # 使用 argmax 找出每個像素點機率最高的類別，得到 [1, H, W] 的預測遮罩
    anomaly_mask_tensor = torch.argmax(seg_map_probs, dim=1)

    # b. 將 Tensor 轉換為可用於顯示的 NumPy Array
    original_image_np = np.array(image.resize((224, 224)))

    # 反正規化重建圖像以便顯示
    recon_image_np = recon_image_tensor.squeeze().cpu().numpy().transpose(1, 2, 0)
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    recon_image_np = std * recon_image_np + mean
    recon_image_np = np.clip(recon_image_np, 0, 1)

    anomaly_mask_np = anomaly_mask_tensor.squeeze().cpu().numpy().astype(np.uint8)

    return original_image_np, recon_image_np, anomaly_mask_np


# --- 使用範例 ---
image_path_to_test = 'path/to/your/test_image.png' # ⬅️ 修改為您要測試的圖片路徑
original, reconstruction, anomaly_mask = predict_anomaly(student_model, image_path_to_test, DEVICE)

# --- 可視化結果 ---
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].imshow(original)
axes[0].set_title('Original Image')
axes[0].axis('off')

axes[1].imshow(reconstruction)
axes[1].set_title('Reconstructed Image')
axes[1].axis('off')

# 將異常遮罩（0和1）與原始圖像疊加顯示
axes[2].imshow(original)
axes[2].imshow(anomaly_mask, cmap='jet', alpha=0.4) # 使用半透明疊加
axes[2].set_title('Anomaly Mask')
axes[2].axis('off')

plt.show()