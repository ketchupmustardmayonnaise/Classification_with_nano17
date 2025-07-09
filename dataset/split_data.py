import os
import csv
import random
'''
data라는 폴더 내에 있는 원본 데이터 파일을, force_estimation / validation / test set으로 나누는 코드
30개의 데이터를 21:6:3 (7:2:1)로 나눔
'''

def split_csv_files(folder_path):
    # 폴더 안의 모든 파일 확인
    for file_name in os.listdir(folder_path):
        # 파일 확장자가 .csv인 경우만 처리
        if file_name.endswith('.csv'):
            file_path = os.path.join(folder_path, file_name)

            # 원본 파일 읽기
            with open(file_path, 'r', encoding='utf-8') as file:
                reader = list(csv.reader(file))

            reader = [lst for lst in reader if lst != '\n' or lst != []]
            # 데이터를 랜덤하게 섞기
            random.shuffle(reader)

            # 전체 개수
            total = len(reader)

            # 7:2:1 비율로 나누기 위한 인덱스 계산
            train_end = int(total * 0.7)
            val_end = train_end + int(total * 0.2)

            # 슬라이싱
            train_set = reader[:train_end]
            val_set = reader[train_end:val_end]
            test_set = reader[val_end:]

            # 새로운 파일 이름 생성
            base_name = os.path.splitext(file_name)[0]
            train_file = os.path.join(folder_path, f"{base_name}_train.csv")
            validation_file = os.path.join(folder_path, f"{base_name}_validation.csv")
            test_file = os.path.join(folder_path, f"{base_name}_test.csv")

            # 새로운 파일로 저장
            def save_csv(data, path):
                with open(path, 'w', encoding='utf-8', newline='') as out_file:
                    writer = csv.writer(out_file)
                    writer.writerows(data)

            save_csv(train_set, train_file)
            save_csv(val_set, validation_file)
            save_csv(test_set, test_file)

            print(f"Processed {file_name}: Saved as {train_file}, {validation_file}, and {test_file}.")


# 특정 폴더 경로 지정
folder_path = "./data/split"
split_csv_files(folder_path)