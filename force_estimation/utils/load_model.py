import os
import wandb

def load_model_from_wandb(model_class, artifact_name, checkpoint_filename, set_eval: bool = True, strict: bool = False):
    print(f"Artifact '{artifact_name}' 다운로드 중...")
    try:
        api = wandb.Api()
        artifact = api.artifact(artifact_name, type='model')
        artifact_dir = artifact.download()
        model_path = os.path.join(artifact_dir, checkpoint_filename)

        if not os.path.exists(model_path): raise FileNotFoundError(f"체크포인트 파일을 찾을 수 없습니다: {model_path}")
        # PyTorch 2.6 changed torch.load default to weights_only=True.
        # This project checkpoints include OmegaConf objects, so use full checkpoint loading.
        model = model_class.load_from_checkpoint(model_path, weights_only=False, strict=strict)
        if set_eval:
            model.eval()
        else:
            model.train()
        print(f"모델 로드 완료: {model_path}")
        return model
    except Exception as e:
        print(f"W&B에서 모델 로드 실패: {e}")
        return None


def load_model_from_local(model_cls, checkpoint_path, set_eval: bool = True, strict: bool = False, **init_kwargs):
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"체크포인트 파일을 찾을 수 없습니다: {checkpoint_path}")

    # PyTorch 2.6 changed torch.load default to weights_only=True.
    # This project checkpoints include OmegaConf objects, so use full checkpoint loading.
    model = model_cls.load_from_checkpoint(
        checkpoint_path,
        **init_kwargs,
        strict=strict,   # strict=True면 키 불일치 즉시 에러
        weights_only=False
    )
    if set_eval:
        model.eval()
    else:
        model.train()
    print(f"모델 로드 완료: {checkpoint_path}")
    return model
