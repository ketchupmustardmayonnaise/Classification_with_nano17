import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

# 1. 모델 정의 (CNN + LSTM)
# ----------------------------------------------------
# 64x64 뎁스맵 시퀀스를 입력받아 힘을 예측하는 모델
class ForceEstimator(nn.Module):
    def __init__(self, cnn_output_dim=128, lstm_hidden_dim=64, num_lstm_layers=1):
        """
        모델의 레이어를 초기화합니다.
        
        Args:
            cnn_output_dim (int): CNN을 통과한 후 특징 벡터의 차원
            lstm_hidden_dim (int): LSTM의 히든 상태 차원
            num_lstm_layers (int): 쌓을 LSTM 레이어의 수
        """
        super(ForceEstimator, self).__init__()

        self.cnn_output_dim = cnn_output_dim
        self.lstm_hidden_dim = lstm_hidden_dim

        # Part 1: 공간 특징 추출기 (Spatial Feature Extractor - CNN)
        # ----------------------------------------------------
        # 입력: (Batch * Seq, 1, 64, 64) -> 출력: (Batch * Seq, cnn_output_dim)
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # -> 16x32x32
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # -> 32x16x16
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # -> 64x8x8
            nn.Flatten(),  # -> 64 * 8 * 8 = 4096
        )
        
        # CNN의 최종 출력을 cnn_output_dim으로 맞춰주는 FC 레이어
        self.fc_after_cnn = nn.Linear(64 * 8 * 8, self.cnn_output_dim)


        # Part 2: 시간 특징 모델링 (Temporal Feature Modeler - LSTM)
        # ----------------------------------------------------
        # 입력: (Batch, Seq, cnn_output_dim) -> 출력: (Batch, Seq, lstm_hidden_dim)
        self.lstm = nn.LSTM(
            input_size=self.cnn_output_dim,
            hidden_size=self.lstm_hidden_dim,
            num_layers=num_lstm_layers,
            batch_first=True  # 입력 텐서의 차원을 (batch, seq, feature)로 설정
        )

        # Part 3: 최종 힘 예측 (Final Force Prediction)
        # ----------------------------------------------------
        # LSTM의 출력을 받아 최종 힘(1개의 값)을 예측하는 FC 레이어
        self.fc_final = nn.Linear(self.lstm_hidden_dim, 1)


    def forward(self, x):
        """
        순전파 로직을 정의합니다.
        x: (batch_size, sequence_length, channels, height, width)
           e.g., (32, 10, 1, 64, 64)
        """
        batch_size, seq_length, c, h, w = x.size()

        # (B, T, C, H, W) -> (B * T, C, H, W)
        # CNN은 시퀀스 개념이 없으므로, 배치와 시퀀스 차원을 하나로 합쳐서 처리
        c_in = x.view(batch_size * seq_length, c, h, w)

        # CNN을 통과시켜 공간 특징 추출
        c_out = self.cnn(c_in)
        
        # 특징 벡터 생성
        feature_vector = self.fc_after_cnn(c_out)

        # (B * T, cnn_output_dim) -> (B, T, cnn_output_dim)
        # LSTM에 입력하기 위해 다시 시퀀스 형태로 복원
        r_in = feature_vector.view(batch_size, seq_length, -1)

        # LSTM을 통과시켜 시간 특징 추출
        # outputs: 모든 시점(time step)의 hidden state
        # h_n: 마지막 시점의 hidden state
        outputs, (h_n, c_n) = self.lstm(r_in)

        # 마지막 시점의 출력(압력이 모두 가해진 후)만 사용하여 힘을 예측
        last_time_step_output = outputs[:, -1, :]
        
        # 최종 힘 예측
        prediction = self.fc_final(last_time_step_output)

        return prediction


# 2. 모델 학습 함수
# ----------------------------------------------------
def train_model(model, train_loader, criterion, optimizer, num_epochs=10, device='cpu'):
    model.to(device)
    
    for epoch in range(num_epochs):
        model.train()  # 모델을 학습 모드로 설정
        running_loss = 0.0

        for i, (sequences, forces) in enumerate(train_loader):
            sequences = sequences.to(device)
            forces = forces.to(device)
            
            # 경사 초기화
            optimizer.zero_grad()
            
            # 모델 순전파
            outputs = model(sequences)
            
            # 손실 계산 (outputs과 forces의 차원을 맞춰주어야 함)
            loss = criterion(outputs.squeeze(), forces)
            
            # 역전파 및 가중치 업데이트
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()

        epoch_loss = running_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}")
    
    print("학습 완료!")


# 3. 실행 예제
# ----------------------------------------------------
if __name__ == "__main__":
    # --- 하이퍼파라미터 설정 ---
    BATCH_SIZE = 32
    SEQ_LENGTH = 10  # 10개의 연속된 뎁스맵을 하나의 시퀀스로 사용
    IMG_SIZE = 64
    LEARNING_RATE = 0.001
    NUM_EPOCHS = 5

    # --- 장치 설정 ---
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # --- 가상 데이터 생성 ---
    # 실제 프로젝트에서는 이 부분을 실제 데이터로 교체해야 합니다.
    # 100개의 샘플, 각 샘플은 10개의 64x64 뎁스맵으로 구성
    num_samples = 100
    dummy_sequences = torch.randn(num_samples, SEQ_LENGTH, 1, IMG_SIZE, IMG_SIZE)
    # 각 시퀀스의 마지막에 해당하는 힘 값 (100개)
    dummy_forces = torch.rand(num_samples) * 10  # 0~10 사이의 힘

    print(f"입력 데이터 형태: {dummy_sequences.shape}") # (100, 10, 1, 64, 64)
    print(f"정답 데이터 형태: {dummy_forces.shape}")   # (100)

    # --- 데이터로더 생성 ---
    train_dataset = TensorDataset(dummy_sequences, dummy_forces)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # --- 모델, 손실 함수, 옵티마이저 초기화 ---
    model = ForceEstimator()
    criterion = nn.MSELoss() # 평균 제곱 오차 (회귀 문제에 적합)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # --- 모델 학습 실행 ---
    train_model(model, train_loader, criterion, optimizer, num_epochs=NUM_EPOCHS, device=device)

    # --- 학습된 모델로 추론 테스트 ---
    model.eval() # 모델을 평가 모드로 설정
    with torch.no_grad():
        # 테스트용 샘플 하나를 가져옴 (B, T, C, H, W)
        sample_sequence = dummy_sequences[0].unsqueeze(0).to(device)
        predicted_force = model(sample_sequence)
        
        print("\n--- 추론 테스트 ---")
        print(f"테스트 시퀀스 입력 형태: {sample_sequence.shape}")
        print(f"예측된 힘: {predicted_force.item():.4f}")
        print(f"실제 힘: {dummy_forces[0].item():.4f}")