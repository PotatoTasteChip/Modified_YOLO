import os

# --- 설정 ---
base_folder = "TP_FP_FN_Classifier/V8_0_542"
output_file = "combined_results.txt"

# --- 클래스 별 텍스트 파일 통합 ---
class_types = ["FP", "FN", "TP"]  # 처리할 폴더 이름
class_names = [
    "hand", "Cider", "Cola", "LetBe", "MoGu", "Narand", "POCARI",
    "POWERADE", "Buldak", "Carbo_Bul", "Doshirak", "Jinla", "Jjapa",
    "Kimchi_Sa", "Neoguri", "Sinla", "Wangtt", "Yukgae", "Picnic",
    "Cocop", "POWERO2"
]

# 결과 저장 파일 초기화
with open(output_file, "w") as outfile:
    outfile.write("Class, FP, FN, TP\n")

# 각 클래스에 대해 FP, FN, TP 합치기
for class_name in class_names:
    fp_count, fn_count, tp_count = 0, 0, 0

    for class_type in class_types:
        folder_path = os.path.join(base_folder, class_type, class_name)
        count_file = os.path.join(folder_path, f"{class_name}_{class_type}_count.txt")

        # 파일이 존재하면 값 읽기
        if os.path.exists(count_file):
            with open(count_file, "r") as f:
                count = int(f.read().strip())
                if class_type == "FP":
                    fp_count = count
                elif class_type == "FN":
                    fn_count = count
                elif class_type == "TP":
                    tp_count = count

    # 결과를 출력 파일에 추가
    with open(output_file, "a") as outfile:
        outfile.write(f"{class_name}, {fp_count}, {fn_count}, {tp_count}\n")

print(f"Combined results saved to {output_file}")
