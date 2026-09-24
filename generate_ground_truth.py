"""Generate structured Ground Truth for sensitive regions (PII - Faces) across all dataset images.

Fulfills C5 Challenge Requirements:
1. Ground truth sensitive regions / identity signal
2. Slice analysis categorization (Small/Hard faces vs Large/Standard faces)
"""

import json
import os
from pathlib import Path
import cv2


def create_ground_truth():
    det_file = Path("detection_results.json")
    dataset_dir = Path("dataset")

    if not det_file.exists():
        raise FileNotFoundError("detection_results.json not found!")

    with open(det_file, "r", encoding="utf-8") as f:
        detections = json.load(f)

    gt_data = {
        "dataset_name": "Team06-C5 PII Release Guard Dataset",
        "description": "Ground truth sensitive regions (Faces/PII) for Privacy-Utility evaluation",
        "total_images": len(detections),
        "total_sensitive_instances": 0,
        "slice_summary": {
            "large_faces": 0,
            "small_faces": 0,
        },
        "images": {},
    }

    face_counter = 0

    for item in detections:
        img_name = item["image"]
        img_path = dataset_dir / img_name
        
        # Read image to obtain exact dimensions
        if img_path.exists():
            img = cv2.imread(str(img_path))
            h, w = img.shape[:2]
        else:
            w, h = 0, 0

        image_entry = {
            "image_name": img_name,
            "width": w,
            "height": h,
            "num_sensitive_regions": len(item.get("detections", [])),
            "sensitive_regions": [],
        }

        for det in item.get("detections", []):
            face_counter += 1
            x1, y1, x2, y2 = det["box_xyxy"]
            
            # Clip bounds
            x1 = max(0.0, min(float(w), x1))
            y1 = max(0.0, min(float(h), y1))
            x2 = max(0.0, min(float(w), x2))
            y2 = max(0.0, min(float(h), y2))
            
            box_w = round(x2 - x1, 2)
            box_h = round(y2 - y1, 2)
            area = round(box_w * box_h, 2)
            
            # Slice categorization: Small (<40px min dimension) vs Large
            is_small = min(box_w, box_h) < 40
            slice_type = "small_distant_face" if is_small else "standard_large_face"
            
            if is_small:
                gt_data["slice_summary"]["small_faces"] += 1
            else:
                gt_data["slice_summary"]["large_faces"] += 1

            region = {
                "instance_id": f"PII_FACE_{face_counter:04d}",
                "type": "face",
                "box_xyxy": [round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)],
                "box_xywh": [round(x1, 2), round(y1, 2), box_w, box_h],
                "area": area,
                "slice": slice_type,
                "confidence_source": det.get("confidence", 1.0),
                "is_difficult": is_small,
            }
            image_entry["sensitive_regions"].append(region)

        gt_data["images"][img_name] = image_entry

    gt_data["total_sensitive_instances"] = face_counter

    # Save to ground_truth.json
    out_path_1 = Path("ground_truth.json")
    with open(out_path_1, "w", encoding="utf-8") as f:
        json.dump(gt_data, f, indent=4, ensure_ascii=False)

    # Also save to ground_truth/sensitive_annotations.json
    gt_dir = Path("ground_truth")
    gt_dir.mkdir(exist_ok=True)
    out_path_2 = gt_dir / "sensitive_annotations.json"
    with open(out_path_2, "w", encoding="utf-8") as f:
        json.dump(gt_data, f, indent=4, ensure_ascii=False)

    print(f"[+] Ground Truth created successfully!")
    print(f"    - Total Images: {gt_data['total_images']}")
    print(f"    - Total Sensitive Faces: {gt_data['total_sensitive_instances']}")
    print(f"    - Standard/Large Faces (>=40px): {gt_data['slice_summary']['large_faces']}")
    print(f"    - Small/Distant Faces (<40px): {gt_data['slice_summary']['small_faces']}")
    print(f"    - Saved to: {out_path_1} and {out_path_2}")


if __name__ == "__main__":
    create_ground_truth()
