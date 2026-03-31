"""
生成多難度等級的合成異常測試集，用於驗證學生模型的泛化能力。
四個難度等級透過「透明度 × 尺寸」控制：
  Level 1 (Obvious):  不透明、大尺寸 — 基線，模型理應輕鬆偵測
  Level 2 (Moderate): 半透明、中等尺寸 — 對模型有一定挑戰
  Level 3 (Subtle):   低透明度、小尺寸 — 考驗模型的敏感度
  Level 4 (Extreme):  極低透明度、極小尺寸 — 接近人眼辨識極限

用法:
    python generate_regular_testset.py [--mvtec_root ./mvtec] [--output_root ./regular_testset] [--obj_id -1]
"""
import os
import cv2
import numpy as np
import argparse
import random


# =======================
# 六種規則形狀繪製函式
# =======================
def draw_square(mask, img, color, cx, cy, size):
    half = size // 2
    cv2.rectangle(img, (cx - half, cy - half), (cx + half, cy + half), color, -1)
    cv2.rectangle(mask, (cx - half, cy - half), (cx + half, cy + half), 255, -1)


def draw_circle(mask, img, color, cx, cy, size):
    radius = size // 2
    cv2.circle(img, (cx, cy), radius, color, -1)
    cv2.circle(mask, (cx, cy), radius, 255, -1)


def draw_triangle(mask, img, color, cx, cy, size):
    half = size // 2
    pts = np.array([
        [cx, cy - half],
        [cx - half, cy + half],
        [cx + half, cy + half]
    ], dtype=np.int32)
    cv2.fillPoly(img, [pts], color)
    cv2.fillPoly(mask, [pts], 255)


def draw_diamond(mask, img, color, cx, cy, size):
    half = size // 2
    pts = np.array([
        [cx, cy - half],
        [cx + half, cy],
        [cx, cy + half],
        [cx - half, cy]
    ], dtype=np.int32)
    cv2.fillPoly(img, [pts], color)
    cv2.fillPoly(mask, [pts], 255)


def draw_star(mask, img, color, cx, cy, size):
    """五角星"""
    outer_r = size // 2
    inner_r = outer_r * 0.4
    pts = []
    for i in range(5):
        angle_outer = np.radians(-90 + i * 72)
        pts.append([int(cx + outer_r * np.cos(angle_outer)),
                     int(cy + outer_r * np.sin(angle_outer))])
        angle_inner = np.radians(-90 + i * 72 + 36)
        pts.append([int(cx + inner_r * np.cos(angle_inner)),
                     int(cy + inner_r * np.sin(angle_inner))])
    pts = np.array(pts, dtype=np.int32)
    cv2.fillPoly(img, [pts], color)
    cv2.fillPoly(mask, [pts], 255)


def draw_cross(mask, img, color, cx, cy, size):
    half = size // 2
    arm = size // 6
    cv2.rectangle(img, (cx - half, cy - arm), (cx + half, cy + arm), color, -1)
    cv2.rectangle(mask, (cx - half, cy - arm), (cx + half, cy + arm), 255, -1)
    cv2.rectangle(img, (cx - arm, cy - half), (cx + arm, cy + half), color, -1)
    cv2.rectangle(mask, (cx - arm, cy - half), (cx + arm, cy + half), 255, -1)


SHAPES = {
    'square': draw_square, 'circle': draw_circle, 'triangle': draw_triangle,
    'diamond': draw_diamond, 'star': draw_star, 'cross': draw_cross,
}

# 六種規則顏色 (BGR for OpenCV)
COLORS = {
    'red': (0, 0, 255), 'green': (0, 255, 0), 'blue': (255, 0, 0),
    'yellow': (0, 255, 255), 'cyan': (255, 255, 0), 'magenta': (255, 0, 255),
}

IMG_DIM = 256

# =======================
# 難度等級配置
# （透過透明度和尺寸的梯度變化來測試模型的泛化極限）
# =======================
DIFFICULTY_LEVELS = {
    'level_1_obvious': {
        'opacity_range': (0.85, 1.0),
        'size_range': (40, 70),
        'desc': '高對比度、大尺寸（基線）'
    },
    'level_2_moderate': {
        'opacity_range': (0.40, 0.60),
        'size_range': (25, 45),
        'desc': '中等透明度、中等尺寸'
    },
    'level_3_subtle': {
        'opacity_range': (0.15, 0.30),
        'size_range': (15, 30),
        'desc': '低透明度、小尺寸'
    },
    'level_4_extreme': {
        'opacity_range': (0.05, 0.12),
        'size_range': (8, 20),
        'desc': '極低透明度、極小尺寸'
    },
}


def apply_shape_with_opacity(bg_img, mask, draw_fn, color, cx, cy, size, opacity):
    """
    以指定透明度在圖片上繪製形狀（alpha blending）。
    透明度越低 → 異常越不明顯 → 越難偵測。
    GT mask 保持二值，不受透明度影響。
    """
    # 在臨時畫布上繪製完整形狀
    overlay = bg_img.copy()
    temp_mask = np.zeros(bg_img.shape[:2], dtype=np.uint8)
    draw_fn(temp_mask, overlay, color, cx, cy, size)

    # Alpha blending：只在形狀區域內混合
    shape_pixels = temp_mask > 0
    bg_img[shape_pixels] = np.clip(
        opacity * overlay[shape_pixels].astype(np.float32) +
        (1 - opacity) * bg_img[shape_pixels].astype(np.float32),
        0, 255
    ).astype(np.uint8)

    # GT mask 保持二值（表示異常的真實位置，不論可見程度）
    mask[shape_pixels] = 255


def generate_for_object(obj_name, mvtec_root, output_root, seed=42):
    """對單個 obj 生成多難度等級的合成異常圖 + mask"""
    rng = random.Random(seed)
    np_rng = np.random.RandomState(seed)

    good_dir = os.path.join(mvtec_root, obj_name, 'train', 'good')
    if not os.path.isdir(good_dir):
        print(f"⚠️ 跳過 {obj_name}: 找不到 {good_dir}")
        return 0

    good_images = sorted([f for f in os.listdir(good_dir) if f.endswith('.png')])
    if len(good_images) == 0:
        print(f"⚠️ 跳過 {obj_name}: 無 good 圖片")
        return 0

    # 複製 good 圖做對照（取 10 張，提高 AUROC 統計品質）
    out_good_dir = os.path.join(output_root, obj_name, 'test', 'good')
    os.makedirs(out_good_dir, exist_ok=True)
    good_count = min(10, len(good_images))
    for i in range(good_count):
        src = cv2.imread(os.path.join(good_dir, good_images[i]))
        src = cv2.resize(src, (IMG_DIM, IMG_DIM))
        cv2.imwrite(os.path.join(out_good_dir, f'{i:03d}.png'), src)

    total_count = 0
    shape_names = list(SHAPES.keys())
    color_names = list(COLORS.keys())

    for level_name, level_config in DIFFICULTY_LEVELS.items():
        out_img_dir = os.path.join(output_root, obj_name, 'test', level_name)
        out_mask_dir = os.path.join(output_root, obj_name, 'ground_truth', level_name)
        os.makedirs(out_img_dir, exist_ok=True)
        os.makedirs(out_mask_dir, exist_ok=True)

        opacity_lo, opacity_hi = level_config['opacity_range']
        size_lo, size_hi = level_config['size_range']
        count = 0

        for shape_name in shape_names:
            for color_name in color_names:
                bg_file = rng.choice(good_images)
                bg_img = cv2.imread(os.path.join(good_dir, bg_file))
                bg_img = cv2.resize(bg_img, (IMG_DIM, IMG_DIM))
                mask = np.zeros((IMG_DIM, IMG_DIM), dtype=np.uint8)

                size = np_rng.randint(size_lo, size_hi + 1)
                margin = size // 2 + 5
                cx = np_rng.randint(margin, IMG_DIM - margin)
                cy = np_rng.randint(margin, IMG_DIM - margin)
                opacity = np_rng.uniform(opacity_lo, opacity_hi)

                color_bgr = COLORS[color_name]
                draw_fn = SHAPES[shape_name]
                apply_shape_with_opacity(bg_img, mask, draw_fn, color_bgr, cx, cy, size, opacity)

                fname = f'{shape_name}_{color_name}_{count:03d}.png'
                mask_fname = f'{shape_name}_{color_name}_{count:03d}_mask.png'
                cv2.imwrite(os.path.join(out_img_dir, fname), bg_img)
                cv2.imwrite(os.path.join(out_mask_dir, mask_fname), mask)
                count += 1

        print(f"  📁 {level_name}: {count} 張 ({level_config['desc']})")
        total_count += count

    print(f"✅ {obj_name}: 生成 {total_count} 張異常圖 + {good_count} 張 good 圖")
    return total_count


def main():
    parser = argparse.ArgumentParser(description='生成多難度等級合成測試集（泛化能力驗證）')
    parser.add_argument('--mvtec_root', type=str, default='./mvtec')
    parser.add_argument('--output_root', type=str, default='./regular_testset')
    parser.add_argument('--obj_id', type=int, default=-1, help='物件 ID (0-14)，-1 表示全部')
    args = parser.parse_args()

    obj_list = [
        'capsule', 'bottle', 'carpet', 'leather', 'pill', 'transistor',
        'tile', 'cable', 'zipper', 'toothbrush', 'metal_nut', 'hazelnut',
        'screw', 'grid', 'wood'
    ]

    if args.obj_id == -1:
        picked = obj_list
    else:
        picked = [obj_list[args.obj_id]]

    total = 0
    for obj_name in picked:
        total += generate_for_object(obj_name, args.mvtec_root, args.output_root)

    print(f"\n🎉 共生成 {total} 張合成異常測試圖（4 個難度等級），輸出至: {args.output_root}")


if __name__ == '__main__':
    main()
