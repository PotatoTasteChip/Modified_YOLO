import torch
import os
from ultralytics import YOLO

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # GPU1만 사용
os.environ["TORCH_CUDNN_ENABLED"] = "1"  # 활성화된 경우에도 유지
os.environ["TORCH_DTYPE"] = "float32"    # 강제 Full Precision
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


if __name__ == '__main__':
    # GPU 설정 (여러 GPU 사용 가능)
    device = '1'  # 0번과 1번 GPU를 사용하도록 명시적으로 지정


    model = YOLO('/home/vprism_3/modified_YOLO5/ultralytics/cfg/models/11/yolo11-fpsa.yaml').load('yolo11x.pt')
    worker = len(os.sched_getaffinity(0)) - 1

    
    # 데이터 증강 설정 추가
    results = model.train(
        data='/home/vprism_3/modified_YOLO5/test_code/data_yaml/B_21.yaml',
        epochs=1500,
        batch=8,
        imgsz=640,
        workers=8,
        device=[1],           # 여러 GPU 지정
        amp=False,                # Mixed Precision (AMP) 활성화
        save=True,
        save_period=5,
        project='runs/train',    # TensorBoard 로그가 저장될 디렉터리 경로 지정
        name='yolo_vprism_5_20240115',          # 실험 이름 지정 (해당 디렉터리에 저장됨)
        flipud=0.0,              # 수직 반전 확률
        fliplr=0.5,              # 수평 반전 확률
        degrees=10.0,            # 회전 각도 범위
        scale=0.5,               # 확대/축소 비율
        translate=0.1,           # 이동 비율
        shear=2.0,               # 왜곡 비율
        hsv_h=0.015,             # 색상 변화 범위
        hsv_s=0.7,               # 채도 변화 범위
        hsv_v=0.4,               # 밝기 변화 범위
        patience=100  # Early Stopping을 위한 patience 설정 (예: 50 에포크)
    )