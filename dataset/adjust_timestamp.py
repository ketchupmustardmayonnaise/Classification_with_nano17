import os
import zipfile
import numpy as np
import io


def process_all_zips(input_root, output_root):
    """
    input_root 내의 모든 zip 파일을 찾아 img_ts를 보정하고,
    output_root에 동일한 구조로 저장합니다.
    """
    for root, dirs, files in os.walk(input_root):
        for file in files:
            if file.lower().endswith('.zip'):
                # 1. 경로 설정
                # 원본 zip의 전체 경로
                original_zip_path = os.path.join(root, file)

                # output_root 내에 생성될 상대 경로 계산 및 폴더 생성
                relative_path = os.path.relpath(root, input_root)
                target_dir = os.path.join(output_root, relative_path)
                os.makedirs(target_dir, exist_ok=True)

                target_zip_path = os.path.join(target_dir, file)

                print(f"처리 중: {original_zip_path}")

                # 2. ZIP 파일 수정 작업
                try:
                    fix_single_zip(original_zip_path, target_zip_path)
                    print(f"저장 완료: {target_zip_path}")
                except Exception as e:
                    print(f"오류 발생 ({file}): {e}")


def fix_single_zip(zip_path, output_path):
    """단일 ZIP 파일 내의 모든 NPZ를 수정하여 새 ZIP으로 저장"""
    with zipfile.ZipFile(zip_path, 'r') as zin, \
            zipfile.ZipFile(output_path, 'w') as zout:

        for item in zin.infolist():
            filename = item.filename
            data = zin.read(filename)

            if filename.endswith('.npz'):
                # NPZ 파일 로드
                with np.load(io.BytesIO(data)) as npz_data:
                    d = dict(npz_data)
                    # img_ts 배열 복사 및 수정
                    img_ts = d['img_ts'].copy()

                    # 보정 로직: diff가 10 이상이면 이전값 + 0.02
                    for i in range(1, len(img_ts)):
                        diff = img_ts[i] - img_ts[i - 1]
                        if diff >= 10:
                            img_ts[i] = img_ts[i - 1] + 0.02

                    d['img_ts'] = img_ts

                    # 메모리 내에서 다시 NPZ로 패킹
                    new_npz_buffer = io.BytesIO()
                    np.savez(new_npz_buffer, **d)
                    new_npz_buffer.seek(0)

                    zout.writestr(filename, new_npz_buffer.read())
            else:
                # npz가 아닌 일반 파일은 그대로 복사
                zout.writestr(item, data)


# --- 사용 설정 ---
# 1. 원본 데이터가 들어있는 루트 폴더
input_folder = './original_data/bodypart'
# 2. 결과물이 저장될 새로운 루트 폴더
output_folder = './original_data/bodypart_adjusted'

process_all_zips(input_folder, output_folder)