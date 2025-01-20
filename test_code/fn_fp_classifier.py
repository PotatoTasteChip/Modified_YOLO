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
def load_labels(label_path):
    """
    정답 라벨 파일을 읽어 객체 정보를 반환.
    라벨 파일 한 줄 형식: class_id center_x center_y width height (좌표: 0~1 정규화)
    """
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
    """
    정규화된 bbox (center_x, center_y, width, height)를
    좌측 상단, 우측 하단 좌표 [x1, y1, x2, y2]로 변환.
    """
    cx, cy, w, h = norm_bbox
    x1 = (cx - w / 2) * img_width
    y1 = (cy - h / 2) * img_height
    x2 = (cx + w / 2) * img_width
    y2 = (cy + h / 2) * img_height
    return np.array([x1, y1, x2, y2])

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

# --- 경로 및 결과 저장 폴더 설정 ---
images_folder = "/home/vprism1/beaver_ws/dataset/main_data/val_3/img"    # 이미지 폴더
labels_folder = "/home/vprism1/beaver_ws/dataset/main_data/val_3/label"    # 정답 라벨 폴더

# 결과 저장 기본 폴더 (클래스별 폴더 내 GT, FN, FP를 개별 저장)
output_base_folder = "FP_FN_results_8x_542_4"
gt_output_folder = os.path.join(output_base_folder, "GT")
fn_output_folder = os.path.join(output_base_folder, "FN")
fp_output_folder = os.path.join(output_base_folder, "FP")
# 추가로, GT_FN과 GT_FP용 별도 폴더 (여기서는 FN/FP 박스를 제외한, 모든 GT 박스만 그린 이미지)
gt_fn_output_folder = os.path.join(output_base_folder, "GT_FN")
gt_fp_output_folder = os.path.join(output_base_folder, "GT_FP")
tp_output_folder = os.path.join(output_base_folder, "TP")

os.makedirs(tp_output_folder, exist_ok=True)
os.makedirs(gt_output_folder, exist_ok=True)
os.makedirs(fn_output_folder, exist_ok=True)
os.makedirs(fp_output_folder, exist_ok=True)
os.makedirs(gt_fn_output_folder, exist_ok=True)
os.makedirs(gt_fp_output_folder, exist_ok=True)

# IoU 임계값
iou_threshold = 0.6

# --- YOLO 모델 로드 ---
model = YOLO("/home/vprism1/beaver_ws/dataset/model/best_v8x.pt") 

# --- 이미지 처리 및 예측 ---
image_files = glob.glob(os.path.join(images_folder, "*.jpg"))

for image_path in image_files:
    filename = os.path.basename(image_path)
    label_path = os.path.join(labels_folder, filename.replace(".jpg", ".txt"))
    
    # 이미지 로드 및 크기 정보 획득
    img = cv2.imread(image_path)
    if img is None:
        continue
    img_height, img_width = img.shape[:2]

    # --- 1. 모델 추론 (predict) ---
    results = model.predict(image_path, conf=0.542, iou=0.6)
    result = results[0]
    
    # 예측 결과에서 바운딩 박스 추출 (xyxy 형식, confidence, class 값 포함)
    pred_objects = []
    if result.boxes is not None:
        for pred in result.boxes.data.cpu().numpy():
            x1, y1, x2, y2, conf, cls = pred
            pred_objects.append({
                'class_id': int(cls),
                'bbox': np.array([x1, y1, x2, y2]),
                'conf': conf
            })
    
    # --- 2. 정답 라벨 로드 및 좌표 변환 ---
    gt_objects = []
    if os.path.exists(label_path):
        gt_raw = load_labels(label_path)
        for gt in gt_raw:
            bbox = convert_bbox(gt['bbox'], img_width, img_height)
            gt_objects.append({
                'class_id': gt['class_id'],
                'bbox': bbox
            })
    
    # --- 3. 예측 객체와 정답 객체 매칭 (클래스 및 IoU 기준) ---
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
                break   # 한 정답에 대해 하나의 예측만 매칭
    
    # --- 4. FN 및 FP 결정 ---
    # FN: 정답 객체 중 매칭되지 않은 경우
    fns = [gt_objects[i] for i in range(len(gt_objects)) if i not in matched_gt]
    # FP: 예측 객체 중 매칭되지 않은 경우
    fps = [pred_objects[i] for i in range(len(pred_objects)) if i not in matched_pred]
    
    tps = [{
        'gt': gt_objects[g_idx],
        'pred': pred_objects[p_idx]
    } for g_idx, p_idx in zip(matched_gt, matched_pred)]
        
    # --- 5. 클래스별로 분리하여 이미지 생성 (GT, FN, FP) ---
    # GT 이미지 (해당 클래스의 모든 정답 박스만 그리기)
    gt_by_class = {}
    for gt in gt_objects:
        gt_by_class.setdefault(gt['class_id'], []).append(gt)
    
    # FN 이미지 (해당 클래스의 놓친 정답 박스만 그리기)
    fn_by_class = {}
    for fn in fns:
        fn_by_class.setdefault(fn['class_id'], []).append(fn)
    
    # FP 이미지 (해당 클래스의 잘못 예측한 박스만 그리기)
    fp_by_class = {}
    for fp in fps:
        fp_by_class.setdefault(fp['class_id'], []).append(fp)
    
    # --- 6. 클래스별 GT 이미지 저장 ---
    for cls, gt_list in gt_by_class.items():
        gt_img = img.copy()
        for gt in gt_list:
            x1, y1, x2, y2 = map(int, gt['bbox'])
            class_name = class_names.get(gt['class_id'], str(gt['class_id']))
            cv2.rectangle(gt_img, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(gt_img, f"GT:{class_name}", (x1, y2 + 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)
        cls_name = class_names.get(cls, str(cls))
        cls_folder = os.path.join(gt_output_folder, cls_name)
        os.makedirs(cls_folder, exist_ok=True)
        gt_filename = filename.replace(".jpg", f"_{cls_name}_GT.jpg")
        cv2.imwrite(os.path.join(cls_folder, gt_filename), gt_img)
    
    # --- 7. 클래스별 FN 이미지 저장 (단, FN 박스만 그린 이미지)
    for cls, fn_list in fn_by_class.items():
        fn_img = img.copy()
        for fn in fn_list:
            x1, y1, x2, y2 = map(int, fn['bbox'])
            class_name = class_names.get(fn['class_id'], str(fn['class_id']))
            cv2.rectangle(fn_img, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(fn_img, f"FN:{class_name}", (x1, y2 + 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)
        cls_name = class_names.get(cls, str(cls))
        cls_folder = os.path.join(fn_output_folder, cls_name)
        os.makedirs(cls_folder, exist_ok=True)
        fn_filename = filename.replace(".jpg", f"_{cls_name}_FN.jpg")
        cv2.imwrite(os.path.join(cls_folder, fn_filename), fn_img)
    
    # --- 8. 클래스별 FP 이미지 저장 (단, FP 박스만 그린 이미지)
    for cls, fp_list in fp_by_class.items():
        fp_img = img.copy()
        for fp in fp_list:
            x1, y1, x2, y2 = map(int, fp['bbox'])
            class_name = class_names.get(fp['class_id'], str(fp['class_id']))
            cv2.rectangle(fp_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(fp_img, f"FP:{class_name}", (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
        cls_name = class_names.get(cls, str(cls))
        cls_folder = os.path.join(fp_output_folder, cls_name)
        os.makedirs(cls_folder, exist_ok=True)
        fp_filename = filename.replace(".jpg", f"_{cls_name}_FP.jpg")
        cv2.imwrite(os.path.join(cls_folder, fp_filename), fp_img)
    
    # --- 9. 추가로, FN 및 FP 이미지에 대해 'GT_FN'과 'GT_FP' 이미지 저장 ---
    # 수정: 이제 GT_FN과 GT_FP 이미지에는 FN 및 FP 박스를 그리지 않고,
    #       오직 모든 GT 박스만을 그려서 저장함.
    # GT_FN 이미지 저장 (모든 GT 박스만 그린 이미지)
    for cls in gt_by_class.keys():
        gt_fn_img = img.copy()
        for gt in gt_objects:
            x1, y1, x2, y2 = map(int, gt['bbox'])
            gt_class = class_names.get(gt['class_id'], str(gt['class_id']))
            cv2.rectangle(gt_fn_img, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(gt_fn_img, f"GT:{gt_class}", (x1, y2+15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)
        cls_name = class_names.get(cls, str(cls))
        cls_folder = os.path.join(gt_fn_output_folder, cls_name)
        os.makedirs(cls_folder, exist_ok=True)
        gt_fn_filename = filename.replace(".jpg", f"_{cls_name}_GT_FN.jpg")
        cv2.imwrite(os.path.join(cls_folder, gt_fn_filename), gt_fn_img)
    
    # GT_FP 이미지 저장 (모든 GT 박스만 그린 이미지)
    for cls in gt_by_class.keys():
        gt_fp_img = img.copy()
        for gt in gt_objects:
            x1, y1, x2, y2 = map(int, gt['bbox'])
            gt_class = class_names.get(gt['class_id'], str(gt['class_id']))
            cv2.rectangle(gt_fp_img, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(gt_fp_img, f"GT:{gt_class}", (x1, y2+15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)
        cls_name = class_names.get(cls, str(cls))
        cls_folder = os.path.join(gt_fp_output_folder, cls_name)
        os.makedirs(cls_folder, exist_ok=True)
        gt_fp_filename = filename.replace(".jpg", f"_{cls_name}_GT_FP.jpg")
        cv2.imwrite(os.path.join(cls_folder, gt_fp_filename), gt_fp_img)

    for cls in gt_by_class.keys():
        tp_img = img.copy()
        for tp in tps:
            gt = tp['gt']
            pred = tp['pred']
            if gt['class_id'] == cls:
                # GT 박스
                x1, y1, x2, y2 = map(int, gt['bbox'])
                class_name = class_names.get(gt['class_id'], str(gt['class_id']))
                cv2.rectangle(tp_img, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(tp_img, f"GT:{class_name}", (x1, y2 + 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
                
                # 예측 박스
                px1, py1, px2, py2 = map(int, pred['bbox'])
                cv2.rectangle(tp_img, (px1, py1), (px2, py2), (0, 255, 0), 2)
                cv2.putText(tp_img, f"P:{class_name}", (px1, py1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        cls_name = class_names.get(cls, str(cls))
        cls_folder = os.path.join(tp_output_folder, cls_name)
        os.makedirs(cls_folder, exist_ok=True)
        tp_filename = filename.replace(".jpg", f"_{cls_name}_TP.jpg")
        cv2.imwrite(os.path.join(cls_folder, tp_filename), tp_img)        
                            
    # --- (옵션) 전체 결과 시각화 (모든 GT와 예측 박스 함께 표시) ---
    vis_img = img.copy()
    for gt in gt_objects:
        x1, y1, x2, y2 = map(int, gt['bbox'])
        gt_class = class_names.get(gt['class_id'], str(gt['class_id']))
        cv2.rectangle(vis_img, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(vis_img, f"GT:{gt_class}", (x1, y2+15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)
    for pred in pred_objects:
        x1, y1, x2, y2 = map(int, pred['bbox'])
        pred_class = class_names.get(pred['class_id'], str(pred['class_id']))
        cv2.rectangle(vis_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(vis_img, f"P:{pred_class}", (x1, y1-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
    
    cv2.imshow("Evaluation", vis_img)
    if cv2.waitKey(30) & 0xFF == ord("q"):
        break

cv2.destroyAllWindows()
