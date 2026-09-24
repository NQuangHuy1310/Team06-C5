from ultralytics import YOLO
import os
import urllib.request
import json

def main():
    weights_path = 'yolov8-face.pt'
    # Link mới từ HuggingFace (đảm bảo không bị lỗi 404)
    weights_url = 'https://huggingface.co/arnabdhar/YOLOv8-Face-Detection/resolve/main/model.pt'
    
    # Tải model weights chuyên nhận diện khuôn mặt nếu chưa có
    if not os.path.exists(weights_path):
        print("Đang tải model YOLOv8 Face từ HuggingFace (khoảng 22MB)...")
        urllib.request.urlretrieve(weights_url, weights_path)
        print("Tải xong model!")

    # Khởi tạo model YOLOv8 Face
    model = YOLO(weights_path) 
    
    # Chạy detection trên tất cả các ảnh trong thư mục 'dataset/'
    print("Bắt đầu detect khuôn mặt trong các ảnh thuộc thư mục dataset...")
    results = model('dataset/', save=True)
    
    # Lưu kết quả ra file JSON
    all_results = []
    for r in results:
        img_name = os.path.basename(r.path)
        detections = []
        
        # r.boxes chứa danh sách các object phát hiện được
        if r.boxes is not None:
            for box in r.boxes:
                # Box coordinates (x1, y1, x2, y2)
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                
                detections.append({
                    "box_xyxy": [round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)],
                    "confidence": round(conf, 4),
                    "class_id": cls_id
                })
                
        all_results.append({
            "image": img_name,
            "path": r.path,
            "detections": detections
        })
        
    json_path = 'detection_results.json'
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=4)

    print("Hoàn thành! Các ảnh kết quả đã được lưu trong thư mục: runs/detect/predict/")
    print(f"File JSON tổng hợp kết quả đã được lưu tại: {json_path}")

if __name__ == '__main__':
    main()
