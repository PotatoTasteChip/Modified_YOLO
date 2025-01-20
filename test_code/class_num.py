import os
from collections import Counter

def count_yolo_classes(label_folder):
    # 클래스 개수를 저장할 Counter 객체 초기화
    class_counter = Counter()

    # 폴더 내의 모든 .txt 파일 읽기
    for file_name in os.listdir(label_folder):
        if file_name.endswith('.txt'):
            file_path = os.path.join(label_folder, file_name)

            # 파일 내용 읽기
            with open(file_path, 'r') as file:
                for line in file:
                    # YOLO 형식에서 클래스 ID는 각 줄의 첫 번째 값
                    class_id = line.split()[0]
                    class_counter[class_id] += 1

    return class_counter

# 폴더 경로 설정
label_folder = "/home/vprism1/beaver_ws/dataset/main_data/val_3/label"  # 실제 경로로 바꾸세요
label_folder = os.path.expanduser(label_folder)

# 클래스별 개수 출력
class_counts = count_yolo_classes(label_folder)
for class_id, count in sorted(class_counts.items()):
    print(f"Class {class_id}: {count} instances")
