import os
import glob
import numpy as np
import cv2
from ultralytics import YOLO

# --- 클래스 번호와 이름 매핑 ---
class_names = {
    0: "hand",
    1: "Cider",
    2: "Cola",
    3: "LetBe",
    4: "MoGu",
    5: "Narand",
    6: "POCARI",
    7: "POWERADE",
    8: "Buldak",
    9: "Carbo_Bul",
    10: "Doshirak",
    11: "Jinla",
    12: "Jjapa",
    13: "Kimchi_Sa",
    14: "Neoguri",
    15: "Sinla",
    16: "Wangtt",
    17: "Yukgae",
    18: "Picnic",
    19: "Cocop",
    20: "POWERO2"
}

# --- 유틸리티 함수 ---
def calculate_iou(box1, box2):
    """
    두 박스 (box1, box2: [x1, y1, x2, y2] 형식) 사이의 IoU 계산.
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = max(0, box1[2] - box1[0]) * max(0, box1[3] - box1[1])
    area2 = max(0, box2[2] - box2[0]) * max(0, box2[3] - box2[1])
    union_area = area1 + area2 - inter_area
    iou = inter_area / union_area if union_area != 0 else 0
    return iou

# --- 경로 및 설정 ---
model = YOLO("/home/vprism1/beaver_ws/dataset/model/best_v11x.pt")
images_folder = "/home/vprism1/beaver_ws/dataset/main_data/val_3/img"  # 이미지 폴더 경로
labels_folder = "/home/vprism1/beaver_ws/dataset/main_data/val_3/label"  
output_folder_fp = "TP_FP_FN_Classifier/V11_0_401/FP"
os.makedirs(output_folder_fp, exist_ok=True)

# --- IoU 및 클래스별 FP 저장 ---
iou_threshold = 0.5
image_files = glob.glob(os.path.join(images_folder, "*.jpg"))

class_fp_counts = {class_name: 0 for class_name in class_names.values()}

def load_labels(label_path):
    objects = []
    with open(label_path, 'r') as f:
        for line in f.readlines():
            parts = line.strip().split()
            if len(parts) == 5:
                cls, cx, cy, w, h = parts
                objects.append({
                    'class_id': int(cls),
                    'bbox': np.array([float(cx), float(cy), float(w), float(h)])
                })
    return objects

def convert_bbox(norm_bbox, img_width, img_height):
    cx, cy, w, h = norm_bbox
    x1 = (cx - w / 2) * img_width
    y1 = (cy - h / 2) * img_height
    x2 = (cx + w / 2) * img_width
    y2 = (cy + h / 2) * img_height
    return np.array([x1, y1, x2, y2])

for image_path in image_files:
    filename = os.path.basename(image_path)
    label_path = os.path.join(labels_folder, filename.replace(".jpg", ".txt"))

    # 이미지 로드 및 크기 정보
    img = cv2.imread(image_path)
    if img is None:
        continue
    img_height, img_width = img.shape[:2]

    # 모델 추론
    results = model.predict(image_path, conf=0.401, iou=0.6)
    result = results[0]

    pred_objects = []
    if result.boxes is not None:
        for pred in result.boxes.data.cpu().numpy():
            x1, y1, x2, y2, conf, cls = pred
            pred_objects.append({
                'class_id': int(cls),
                'bbox': np.array([x1, y1, x2, y2]),
                'conf': conf
            })

    gt_objects = []
    if os.path.exists(label_path):
        gt_raw = load_labels(label_path)
        for gt in gt_raw:
            bbox = convert_bbox(gt['bbox'], img_width, img_height)
            gt_objects.append({
                'class_id': gt['class_id'],
                'bbox': bbox
            })

    matched_gt = set()
    matched_pred = set()

    for p_idx, pred in enumerate(pred_objects):
        for g_idx, gt in enumerate(gt_objects):
            if g_idx in matched_gt:
                continue
            if pred['class_id'] != gt['class_id']:
                continue
            iou = calculate_iou(pred['bbox'], gt['bbox'])
            if iou >= iou_threshold:
                matched_pred.add(p_idx)
                matched_gt.add(g_idx)

    # FP 계산: 매칭되지 않은 예측값
    for p_idx, pred in enumerate(pred_objects):
        if p_idx not in matched_pred:
            class_name = class_names[pred['class_id']]
            class_fp_counts[class_name] += 1

            # FP 저장 폴더 생성
            class_folder_fp = os.path.join(output_folder_fp, class_name)
            os.makedirs(class_folder_fp, exist_ok=True)

            # FP 이미지 저장
            fp_img = img.copy()
            x1, y1, x2, y2 = map(int, pred['bbox'])
            cv2.rectangle(fp_img, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(fp_img, f"FP:{class_name}", (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            fp_filename = filename.replace(".jpg", f"_{class_name}_FP.jpg")
            cv2.imwrite(os.path.join(class_folder_fp, fp_filename), fp_img)

# FP 개수 기록
for class_name, fp_count in class_fp_counts.items():
    with open(os.path.join(output_folder_fp, f"{class_name}_FP_count.txt"), 'w') as f:
        f.write(f"{fp_count}\n")
