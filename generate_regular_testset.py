"""
生成規則形狀 + 規則顏色的合成異常測試集，用於驗證學生模型的泛化能力。
每個 obj 從 train/good 中隨機取圖片，疊加 6 種形狀 × 6 種顏色 = 36 張異常圖。
同時生成對應的 ground_truth mask。

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
        # 外頂點
        angle_outer = np.radians(-90 + i * 72)
        pts.append([int(cx + outer_r * np.cos(angle_outer)),
                     int(cy + outer_r * np.sin(angle_outer))])
        # 內頂點
        angle_inner = np.radians(-90 + i * 72 + 36)
        pts.append([int(cx + inner_r * np.cos(angle_inner)),
                     int(cy + inner_r * np.sin(angle_inner))])
    pts = np.array(pts, dtype=np.int32)
    cv2.fillPoly(img, [pts], color)
    cv2.fillPoly(mask, [pts], 255)


def draw_cross(mask, img, color, cx, cy, size):
    half = size // 2
    arm = size // 6  # 十字臂寬
    # 水平臂
    cv2.rectangle(img, (cx - half, cy - arm), (cx + half, cy + arm), color, -1)
    cv2.rectangle(mask, (cx - half, cy - arm), (cx + half, cy + arm), 255, -1)
    # 垂直臂
    cv2.rectangle(img, (cx - arm, cy - half), (cx + arm, cy + half), color, -1)
    cv2.rectangle(mask, (cx - arm, cy - half), (cx + arm, cy + half), 255, -1)


SHAPES = {
    'square': draw_square,
    'circle': draw_circle,
    'triangle': draw_triangle,
    'diamond': draw_diamond,
    'star': draw_star,
    'cross': draw_cross,
}

# 六種規則顏色 (BGR for OpenCV)
COLORS = {
    'red': (0, 0, 255),
    'green': (0, 255, 0),
    'blue': (255, 0, 0),
    'yellow': (0, 255, 255),
    'cyan': (255, 255, 0),
    'magenta': (255, 0, 255),
}

IMG_DIM = 256


def generate_for_object(obj_name, mvtec_root, output_root, seed=42):
    """對單個 obj 生成 36 張合成異常圖 + mask"""
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

    # 輸出目錄
    out_img_dir = os.path.join(output_root, obj_name, 'test', 'regular_shape')
    out_mask_dir = os.path.join(output_root, obj_name, 'ground_truth', 'regular_shape')
    # 同時放一份 good 圖片做對照
    out_good_dir = os.path.join(output_root, obj_name, 'test', 'good')
    os.makedirs(out_img_dir, exist_ok=True)
    os.makedirs(out_mask_dir, exist_ok=True)
    os.makedirs(out_good_dir, exist_ok=True)

    # 複製幾張 good 圖做對照 (取 6 張)
    good_count = min(6, len(good_images))
    for i in range(good_count):
        src = cv2.imread(os.path.join(good_dir, good_images[i]))
        src = cv2.resize(src, (IMG_DIM, IMG_DIM))
        cv2.imwrite(os.path.join(out_good_dir, f'{i:03d}.png'), src)

    count = 0
    shape_names = list(SHAPES.keys())
    color_names = list(COLORS.keys())

    for s_idx, shape_name in enumerate(shape_names):
        for c_idx, color_name in enumerate(color_names):
            # 隨機選一張 good 圖作為背景
            bg_file = rng.choice(good_images)
            bg_img = cv2.imread(os.path.join(good_dir, bg_file))
            bg_img = cv2.resize(bg_img, (IMG_DIM, IMG_DIM))

            mask = np.zeros((IMG_DIM, IMG_DIM), dtype=np.uint8)

            # 形狀大小: 隨機 30~60 像素
            size = np_rng.randint(30, 61)
            # 位置: 隨機但確保形狀在圖內
            margin = size // 2 + 5
            cx = np_rng.randint(margin, IMG_DIM - margin)
            cy = np_rng.randint(margin, IMG_DIM - margin)

            color_bgr = COLORS[color_name]
            draw_fn = SHAPES[shape_name]
            draw_fn(mask, bg_img, color_bgr, cx, cy, size)

            fname = f'{shape_name}_{color_name}_{count:03d}.png'
            mask_fname = f'{shape_name}_{color_name}_{count:03d}_mask.png'

            cv2.imwrite(os.path.join(out_img_dir, fname), bg_img)
            cv2.imwrite(os.path.join(out_mask_dir, mask_fname), mask)
            count += 1

    print(f"✅ {obj_name}: 生成 {count} 張異常圖 + {good_count} 張 good 圖")
    return count


def main():
    parser = argparse.ArgumentParser(description='生成規則形狀合成測試集')
    parser.add_argument('--mvtec_root', type=str, default='./mvtec')
    parser.add_argument('--output_root', type=str, default='./regular_testset')
    parser.add_argument('--obj_id', type=int, default=-1,
                        help='物件 ID (0-14)，-1 表示全部')
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

    print(f"\n🎉 共生成 {total} 張合成異常測試圖，輸出至: {args.output_root}")


if __name__ == '__main__':
    main()
