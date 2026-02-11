

def reconstruct_list_from_windows(raw_windows : list,
                                  index : int = None) -> list:
    if index is None:
        index = len(raw_windows[0]) - 1

    reconstructed_list = []
    reconstructed_list.extend(raw_windows[0][:index])  # 첫 번째 데이터는 모두 저장
    for i in range(1, len(raw_windows)):
        reconstructed_list.append(raw_windows[i][index - 1])  # 그 이후부턴 마지막 프레임만 저장