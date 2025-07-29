import numpy as np
import cv2
import torch

# online augnemtation을 적용한다면, 한 시퀀스 내 모든 이미지가 동일하게 flip하거나 회전하여야 함.

# 3차원 이미지들에 대한 코드 (rgb)
def apply_augmentation_to_all_frame(X_batch, angle):
    """
    X_batch: Tensor or ndarray with shape (batch_size, frame, height, width)
    Returns: Tensor with same shape
    """
    batch_size, num_frames, C, H, W = X_batch.shape
    X_out = []

    for b in range(batch_size):
        sample_out = []

        # 하나의 augmentation 파라미터 세트를 전체 batch에 적용
        angle = np.random.uniform(-angle, angle)
        do_vflip = np.random.rand() < 0.5
        do_hflip = np.random.rand() < 0.5

        for f in range(num_frames):
            # (C, H, W) → (H, W, C) for OpenCV
            img = X_batch[b, f].permute(1, 2, 0).cpu().numpy().astype(np.uint8)

            # 회전
            M = cv2.getRotationMatrix2D((W // 2, H // 2), angle, 1)
            img = cv2.warpAffine(img, M, (W, H), borderMode=cv2.BORDER_REPLICATE)
            if len(img.shape) == 2: img = np.expand_dims(img, axis=2)

            # flip
            if do_vflip:
                img = cv2.flip(img, 0)
            if do_hflip:
                img = cv2.flip(img, 1)

            # 이미지가 3차원인 경우에는 괜찮은데, 1차원인 경우에는 shape 유지가 안 되기 때문에
            # 변환 모두 끝난 후 차원을 추가해 줌
            if len(img.shape) == 2: img = np.expand_dims(img, axis=2)

            # (H, W, C) → (C, H, W)
            img = torch.from_numpy(img).permute(2, 0, 1).float()
            sample_out.append(img)

        # stack per-frame tensors into one tensor for this sample
        X_out.append(torch.stack(sample_out))  # shape: (num_frames, C, H, W)

    X_out = torch.stack(X_out).to(dtype=torch.float32)
    return X_out # (batch_size, frame, C, H, W)