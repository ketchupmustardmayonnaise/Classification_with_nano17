import h5py
import numpy as np

# 파일을 '읽기 모드(r)'로 엽니다.
# with 구문을 사용하면 파일을 다 쓴 후 자동으로 닫아줍니다.
with h5py.File('./dataset/test_dataset.h5', 'r') as f:
    
    # 1. 파일의 최상위 그룹/데이터셋 목록 확인 (딕셔너리의 keys()와 유사)
    print("최상위 목록:", list(f.keys()))
    
    # 2. 특정 데이터셋에 접근하고 데이터 읽기
    # 예: 파일 내에 'layer1' 그룹 안의 'weights' 데이터셋이 있다면
    if 'layer1/weights' in f:
        weights_dataset = f['layer1/weights']
        
        # 데이터셋의 모양(shape)과 타입(dtype) 확인
        print("\n'layer1/weights'의 모양:", weights_dataset.shape)
        print("'layer1/weights'의 타입:", weights_dataset.dtype)
        
        # 실제 데이터를 NumPy 배열로 가져오기
        weights_data = weights_dataset[:]  # [:] 슬라이싱을 사용
        # 또는 weights_data = np.array(weights_dataset)
        
        print("실제 데이터 (일부):")
        print(weights_data[0]) # 첫 번째 데이터 출력
        
    # 3. 파일의 모든 구조를 재귀적으로 탐색하는 함수
    def print_h5_structure(group, indent=''):
        for key in group.keys():
            item = group[key]
            if isinstance(item, h5py.Dataset):
                print(f"{indent}Dataset: {key} (shape: {item.shape}, dtype: {item.dtype})")
            elif isinstance(item, h5py.Group):
                print(f"{indent}Group: {key}")
                print_h5_structure(item, indent + '  ') # 들여쓰기 추가

    print("\n--- 전체 파일 구조 ---")
    print_h5_structure(f)