import yaml
import ast
from util import *

class ModelTrainer:
    # 이전 답변에서 제공된 ModelTrainer 클래스 코드를 그대로 여기에 붙여넣으세요.
    def __init__(self, cfg, args):
        # parameters
        self.cfg = cfg
        self.args = args
        self.best_epoch = 0

        # 1. 파라미터 초기화를 위한 헬퍼 메소드 정의
        def _init_param(key, p_type=None):
            value = getattr(self.args, key, None)
            if value is None:
                value = self.cfg.get(key)
            
            if value is not None and p_type is not None:
                # --- 수정된 부분 ---
                # p_type이 ast.literal_eval이고, 실제 값이 문자열일 때만 변환을 수행합니다.
                if p_type is ast.literal_eval:
                    if isinstance(value, str):
                        value = p_type(value)
                else: # int, float, str2bool 등 다른 타입 변환은 그대로 수행
                    value = p_type(value)
                # --- 수정 끝 ---
                    
            setattr(self, key, value)

        # 2. 공통 파라미터 리스트 정의
        common_params = [
            ('train_mode', str2bool), ('use_wandb', str2bool), ('image_resize', str2bool),
            ('target_indices', int), ('stride', int), ('augmentation_angle', int),
            ('num_epoch', int), ('encoder_output', int), ('rnn_layer', int),
            ('decoder', str), ('decoder_output', int), ('fc_num_layer', int),
            ('fc_hidden_dim', int), ('learning_rate', float), ('weight_decay', float),
            ('dropout', float), ('batch_size', int), ('image_type', str),
            ('encoder', str), ('optimizer', str), ('scheduler_type', str),
            ('loss_type', str), ('image_size', ast.literal_eval)
        ]

        # 3. 반복문을 통해 공통 파라미터 한 번에 초기화
        for key, p_type in common_params:
            _init_param(key, p_type)

def main():
    # 1. 커맨드 라인 인자 파서 설정

    args = parse_args()

    # 2. YAML 설정 파일 불러오기
    with open("config.yaml", 'r') as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    # 3. ModelTrainer 인스턴스 생성
    trainer = ModelTrainer(cfg, args)

    # 4. 설정값 확인 (결과 출력)
    print("--- 최종 적용된 설정값 ---")
    print(f"Train Mode: {trainer.train_mode} (Type: {type(trainer.train_mode)})")
    print(f"Number of Epochs: {trainer.num_epoch}")
    print(f"Learning Rate: {trainer.learning_rate}")
    print(f"Batch Size: {trainer.batch_size}")
    print(f"Optimizer: {trainer.optimizer}")
    print(f"Image Size: {trainer.image_size} (Type: {type(trainer.image_size)})")
    print("-------------------------")


if __name__ == '__main__':
    # PyYAML 라이브러리가 없다면 설치해야 합니다.
    # pip install PyYAML
    main()