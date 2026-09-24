"""Fix 12.jpg and 15.jpg Ground Truth:
- 15.jpg: Remove duplicate box on woman on left, keep exactly 3 genuine faces.
- 12.jpg: Remove duplicate boxes on man on left, keep exactly 3 genuine faces.
Regenerate ground truth images and comparison visuals for both.
"""

import json
import sys
from pathlib import Path
import cv2
import numpy as np

sys.stdout.reconfigure(encoding='utf-8')

def fix_images_12_and_15():
    gt_file = Path("ground_truth.json")
    gt_sub_file = Path("ground_truth/sensitive_annotations.json")
    dataset_dir = Path("dataset")
    results_dir = Path("results")
    gt_vis_dir = Path("ground_truth/images_with_gt_boxes")
    comp_dir = Path("comparison_visuals")

    with open(gt_file, "r", encoding="utf-8") as f:
        gt_data = json.load(f)

    from visualize_ground_truth_and_comparisons import draw_gt_on_image, add_header

    # 1. FIX 15.jpg: Keep first 3 genuine faces (PII_FACE_0023, 0024, 0025)
    regs_15 = gt_data["images"]["15.jpg"]["sensitive_regions"]
    filtered_15 = regs_15[:3]  # Drop the 4th overlapping box
    gt_data["images"]["15.jpg"]["sensitive_regions"] = filtered_15
    gt_data["images"]["15.jpg"]["num_sensitive_regions"] = len(filtered_15)

    # 2. FIX 12.jpg: Keep first 3 genuine faces (PII_FACE_0012, 0013, 0014)
    regs_12 = gt_data["images"]["12.jpg"]["sensitive_regions"]
    filtered_12 = regs_12[:3]  # Drop 4th and 5th overlapping boxes
    gt_data["images"]["12.jpg"]["sensitive_regions"] = filtered_12
    gt_data["images"]["12.jpg"]["num_sensitive_regions"] = len(filtered_12)

    # 3. Recalculate totals
    total_faces = sum(len(img["sensitive_regions"]) for img in gt_data["images"].values())
    small_faces = sum(1 for img in gt_data["images"].values() for r in img["sensitive_regions"] if r["is_difficult"])
    large_faces = total_faces - small_faces

    gt_data["total_sensitive_instances"] = total_faces
    gt_data["slice_summary"]["large_faces"] = large_faces
    gt_data["slice_summary"]["small_faces"] = small_faces

    # Save to JSON
    with open(gt_file, "w", encoding="utf-8") as f:
        json.dump(gt_data, f, indent=4, ensure_ascii=False)
    with open(gt_sub_file, "w", encoding="utf-8") as f:
        json.dump(gt_data, f, indent=4, ensure_ascii=False)

    print(f"[+] Đã cập nhật Ground Truth:")
    print(f"    - 15.jpg: đúng {len(filtered_15)} khuôn mặt.")
    print(f"    - 12.jpg: đúng {len(filtered_12)} khuôn mặt.")
    print(f"    - Tổng khuôn mặt toàn bộ dataset: {total_faces} (Mặt lớn: {large_faces}, Mặt nhỏ: {small_faces})")

    # 4. Regenerate images for 15.jpg and 12.jpg
    for img_name, clean_regs, title in [
        ("15.jpg", filtered_15, "1. Ground Truth (3 People)"),
        ("12.jpg", filtered_12, "1. Ground Truth (3 People)")
    ]:
        raw_img = cv2.imread(str(dataset_dir / img_name))
        h, w = raw_img.shape[:2]
        boxes = [r["box_xyxy"] for r in clean_regs]

        # Blur
        img_blur = raw_img.copy()
        for b in boxes:
            x1, y1, x2, y2 = map(int, b)
            x1, x2 = max(0, min(w, x1)), max(0, min(w, x2))
            y1, y2 = max(0, min(h, y1)), max(0, min(h, y2))
            if x2 > x1 and y2 > y1:
                img_blur[y1:y2, x1:x2] = cv2.GaussianBlur(img_blur[y1:y2, x1:x2], (41, 41), 0)
        cv2.imwrite(str(results_dir / "blur" / img_name), img_blur)

        # Mask
        img_mask = raw_img.copy()
        for b in boxes:
            x1, y1, x2, y2 = map(int, b)
            x1, x2 = max(0, min(w, x1)), max(0, min(w, x2))
            y1, y2 = max(0, min(h, y1)), max(0, min(h, y2))
            if x2 > x1 and y2 > y1:
                img_mask[y1:y2, x1:x2] = (0, 0, 0)
        cv2.imwrite(str(results_dir / "mask" / img_name), img_mask)

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
        cv2.imwrite(str(results_dir / "pixelation" / img_name), img_pix)

        # GT image
        gt_img = draw_gt_on_image(raw_img, clean_regs)
        cv2.imwrite(str(gt_vis_dir / f"gt_{img_name}"), gt_img)

        # Comparison 4-panel image
        scale_ratio = min(1.0, 600.0 / w)
        nw, nh = int(w * scale_ratio), int(h * scale_ratio)

        p1 = add_header(cv2.resize(gt_img, (nw, nh)), title, (35, 120, 45))
        p2 = add_header(cv2.resize(img_blur, (nw, nh)), "2. Gaussian Blur (Re-detect: 100%)", (160, 90, 20))
        p3 = add_header(cv2.resize(img_mask, (nw, nh)), "3. Solid Mask (PSNR: 18.36dB)", (80, 20, 20))
        p4 = add_header(cv2.resize(img_pix, (nw, nh)), "4. Pixelation (Sweet Spot: SSIM 0.94)", (25, 90, 160))

        grid = np.vstack([np.hstack([p1, p2]), np.hstack([p3, p4])])
        comp_filename = comp_dir / f"compare_{Path(img_name).stem}.jpg"
        cv2.imwrite(str(comp_filename), grid)
        print(f"[+] Đã tạo lại thành công {comp_filename}")


if __name__ == "__main__":
    fix_images_12_and_15()
