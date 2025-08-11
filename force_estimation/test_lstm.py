import torch
import torch.nn as nn

# LSTM 레이어 정의 (은닉 상태 크기 = 10)
lstm = nn.LSTM(input_size=80, hidden_size=10, bidirectional= True)

# 입력 데이터 생성 (시퀀스 길이=45, 배치 크기=1, 입력 특징=5)
input_seq = torch.randn(45, 1, 80)

# LSTM 실행
outputs, (hidden_state, cell_state) = lstm(input_seq)

print("1. 모든 타임스텝의 출력(outputs) 형태:")
print(outputs.shape)
# 출력: torch.Size([3, 1, 10]) -> 시퀀스 길이 3개에 대한 3개의 출력이 나옴
print(outputs[-1,:,:10])
print(outputs[0,:,10:])
print("\n2. 최종 은닉 상태(hidden_state) 형태:")
print(hidden_state.shape)
# 출력: torch.Size([1, 1, 10]) -> 마지막 타임스텝의 은닉 상태 1개만 나옴
print(hidden_state[0].shape)
print(hidden_state[1].shape)

print("\n3. 최종 셀 상태(cell_state) 형태:")
print(cell_state.shape)
# 출력: torch.Size([1, 1, 10]) -> 마지막 타임스텝의 셀 상태 1개만 나옴