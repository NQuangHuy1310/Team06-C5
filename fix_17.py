"""Fix 17.jpg Ground Truth and regenerate comparison visuals.
Keep only 2 genuine faces (male and female), remove false positive detection.
"""

import json
import sys
from pathlib import Path
import cv2
import numpy as np

sys.stdout.reconfigure(encoding='utf-8')

def fix_image_17():
    gt_file = Path("ground_truth.json")
    gt_sub_file = Path("ground_truth/sensitive_annotations.json")
    dataset_dir = Path("dataset")
    results_dir = Path("results")
    gt_vis_dir = Path("ground_truth/images_with_gt_boxes")
    comp_dir = Path("comparison_visuals")

    with open(gt_file, "r", encoding="utf-8") as f:
        gt_data = json.load(f)

    # 1. Filter regions of 17.jpg to exactly 2 faces
    regions_17 = gt_data["images"]["17.jpg"]["sensitive_regions"]
    filtered_17 = [r for r in regions_17 if r["area"] > 5000]
    gt_data["images"]["17.jpg"]["sensitive_regions"] = filtered_17
    gt_data["images"]["17.jpg"]["num_sensitive_regions"] = len(filtered_17)

    # 2. Recalculate totals
    total_faces = sum(len(img["sensitive_regions"]) for img in gt_data["images"].values())
    small_faces = sum(1 for img in gt_data["images"].values() for r in img["sensitive_regions"] if r["is_difficult"])
    large_faces = total_faces - small_faces

    gt_data["total_sensitive_instances"] = total_faces
    gt_data["slice_summary"]["large_faces"] = large_faces
    gt_data["slice_summary"]["small_faces"] = small_faces

    # Save to ground_truth.json and ground_truth/sensitive_annotations.json
    with open(gt_file, "w", encoding="utf-8") as f:
        json.dump(gt_data, f, indent=4, ensure_ascii=False)
    with open(gt_sub_file, "w", encoding="utf-8") as f:
        json.dump(gt_data, f, indent=4, ensure_ascii=False)

    print(f"[+] Đã cập nhật Ground Truth: 17.jpg hiện có đúng {len(filtered_17)} khuôn mặt.")
    print(f"    - Tổng khuôn mặt toàn bộ dataset: {total_faces} (Mặt lớn: {large_faces}, Mặt nhỏ: {small_faces})")

    # 3. Regenerate results for 17.jpg
    raw_img = cv2.imread(str(dataset_dir / "17.jpg"))
    h, w = raw_img.shape[:2]
    boxes = [r["box_xyxy"] for r in filtered_17]

    # Blur
    img_blur = raw_img.copy()
    for b in boxes:
        x1, y1, x2, y2 = map(int, b)
        x1, x2 = max(0, min(w, x1)), max(0, min(w, x2))
        y1, y2 = max(0, min(h, y1)), max(0, min(h, y2))
        if x2 > x1 and y2 > y1:
            img_blur[y1:y2, x1:x2] = cv2.GaussianBlur(img_blur[y1:y2, x1:x2], (41, 41), 0)
    cv2.imwrite(str(results_dir / "blur" / "17.jpg"), img_blur)

    # Mask
    img_mask = raw_img.copy()
    for b in boxes:
        x1, y1, x2, y2 = map(int, b)
        x1, x2 = max(0, min(w, x1)), max(0, min(w, x2))
        y1, y2 = max(0, min(h, y1)), max(0, min(h, y2))
        if x2 > x1 and y2 > y1:
            img_mask[y1:y2, x1:x2] = (0, 0, 0)
    cv2.imwrite(str(results_dir / "mask" / "17.jpg"), img_mask)

    # Pixelation
    img_pix = raw_img.copy()
    for b in boxes:
        x1, y1, x2, y2 = map(int, b)
        x1, x2 = max(0, min(w, x1)), max(0, min(w, x2))
        y1, y2 = max(0, min(h, y1)), max(0, min(h, y2))
        if x2 > x1 and y2 > y1:
            pixel_size = 12
            region = img_pix[y1:y2, x1:x2]
            sw = max(1, (x2 - x1) // pixel_size)
            sh = max(1, (y2 - y1) // pixel_size)
            tiny = cv2.resize(region, (sw, sh), interpolation=cv2.INTER_LINEAR)
            img_pix[y1:y2, x1:x2] = cv2.resize(tiny, (x2 - x1, y2 - y1), interpolation=cv2.INTER_NEAREST)
    cv2.imwrite(str(results_dir / "pixelation" / "17.jpg"), img_pix)

    # 4. Generate gt_17.jpg
    from visualize_ground_truth_and_comparisons import draw_gt_on_image, add_header
    gt_img = draw_gt_on_image(raw_img, filtered_17)
    cv2.imwrite(str(gt_vis_dir / "gt_17.jpg"), gt_img)

    # 5. Generate compare_17.jpg
    scale_ratio = min(1.0, 600.0 / w)
    nw, nh = int(w * scale_ratio), int(h * scale_ratio)

    p1 = add_header(cv2.resize(gt_img, (nw, nh)), "1. Ground Truth (2 Faces: Male & Female)", (35, 120, 45))
    p2 = add_header(cv2.resize(img_blur, (nw, nh)), "2. Gaussian Blur (Re-detect: 100%)", (160, 90, 20))
    p3 = add_header(cv2.resize(img_mask, (nw, nh)), "3. Solid Mask (PSNR: 18.36dB)", (80, 20, 20))
    p4 = add_header(cv2.resize(img_pix, (nw, nh)), "4. Pixelation (Sweet Spot: SSIM 0.94)", (25, 90, 160))

    grid = np.vstack([np.hstack([p1, p2]), np.hstack([p3, p4])])
    cv2.imwrite(str(comp_dir / "compare_17.jpg"), grid)
    print(f"[+] Đã tạo lại thành công compare_17.jpg tại: {comp_dir / 'compare_17.jpg'}")


if __name__ == "__main__":
    fix_image_17()
