import h5py
import numpy as np

def calculate_force_stats(file_path):
    """
    HDF5 파일 내 모든 시퀀스의 'force' 데이터를 취합하여
    전체 평균과 표준편차를 계산합니다.
    
    Args:
        file_path (str): 분석할 .h5 파일의 경로
        
    Returns:
        tuple: (전체 평균, 전체 표준편차) 튜플
    """
    
    # 각 시퀀스의 'force' 데이터를 저장할 리스트
    all_force_data = []
    
    try:
        with h5py.File(file_path, 'r') as f:
            # 파일 내의 모든 그룹(sequence_xxxx)을 순회
            for group_name in f.keys():
                # 'sequence_'로 시작하는 그룹만 대상으로 함
                if group_name.startswith('sequence_'):
                    # 해당 그룹 내의 'force' 데이터셋에 접근
                    if 'force' in f[group_name]:
                        force_dataset = f[group_name]['force'][:] # NumPy 배열로 데이터 읽기
                        all_force_data.append(force_dataset)
                    else:
                        print(f"경고: '{group_name}' 그룹에 'force' 데이터셋이 없습니다.")
                        
        if not all_force_data:
            print("오류: 파일에서 'force' 데이터를 찾을 수 없습니다.")
            return None, None
            
        # 리스트에 저장된 모든 force 데이터를 하나의 큰 NumPy 배열로 합치기
        # axis=0은 배열들을 세로(행) 방향으로 쌓아 올리라는 의미입니다.
        combined_forces = np.concatenate(all_force_data, axis=0)
        
        # 전체 데이터에 대해 각 열(6개 축)의 평균과 표준편차 계산
        mean_val = np.mean(combined_forces, axis=0)
        std_val = np.std(combined_forces, axis=0)
        
        return mean_val, std_val, combined_forces.shape

    except FileNotFoundError:
        print(f"오류: '{file_path}' 파일을 찾을 수 없습니다.")
        return None, None

# --- 코드 실행 ---
# 여기에 실제 파일 경로를 입력하세요.
h5_file = './dataset/alexnetDiff/test_dataset.h5' 

mean, std, total_shape = calculate_force_stats(h5_file)

if mean is not None and std is not None:
    print(f"✅ 분석 완료!")
    print(f"통합된 'force' 데이터의 전체 Shape: {total_shape}")
    print("-" * 30)
    print(f"📊 전체 Force 평균 (6축):\n{mean}")
    print("-" * 30)
    print(f"📊 전체 Force 표준편차 (6축):\n{std}")