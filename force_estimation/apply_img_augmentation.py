import numpy as np
import cv2
import torch

def apply_all_frames(X_batch):
    """
    X_batch: Tensor or ndarray with shape (batch_size, frame, height, width)
    Returns: Tensor with same shape
    """
    batch_size, num_frames, H, W = X_batch.shape
    X_out = []

    # 하나의 augmentation 파라미터 세트를 전체 batch에 적용
    angle = np.random.uniform(-20, 20)
    do_vflip = np.random.rand() < 0.5
    do_hflip = np.random.rand() < 0.5

    for b in range(batch_size):
        sample_out = []
        for f in range(num_frames):
            img = np.array(X_batch[b, f])

            # 회전
            M = cv2.getRotationMatrix2D((W // 2, H // 2), angle, 1)
            img = cv2.warpAffine(img, M, (W, H), borderMode=cv2.BORDER_REPLICATE)

            # flip
            if do_vflip:
                img = cv2.flip(img, 0)
            if do_hflip:
                img = cv2.flip(img, 1)

            # _img = img.copy()
            # cv2.imshow('img', np.expand_dims(_img, axis=2))
            # cv2.waitKey(1)

            sample_out.append(torch.from_numpy(img).float())

        X_out.append(torch.stack(sample_out))  # (frame, H, W)

    return torch.stack(X_out)  # (batch_size, frame, H, W)


def apply_all_frames_RGB_image(X_batch):
    """
    X_batch: Tensor or ndarray with shape (batch_size, frame, height, width)
    Returns: Tensor with same shape
    """
    batch_size, num_frames, C, H, W = X_batch.shape
    X_out = []

    # 하나의 augmentation 파라미터 세트를 전체 batch에 적용
    angle = np.random.uniform(-180, 180)
    do_vflip = np.random.rand() < 0.5
    do_hflip = np.random.rand() < 0.5

    for b in range(batch_size):
        sample_out = []
        for f in range(num_frames):
            # (C, H, W) → (H, W, C) for OpenCV
            img = X_batch[b, f].permute(1, 2, 0).cpu().numpy().astype(np.uint8)

            # 회전
            M = cv2.getRotationMatrix2D((W // 2, H // 2), angle, 1)
            img = cv2.warpAffine(img, M, (W, H), borderMode=cv2.BORDER_REPLICATE)

            # flip
            if do_vflip:
                img = cv2.flip(img, 0)
            if do_hflip:
                img = cv2.flip(img, 1)

            # (H, W, C) → (C, H, W)
            img = torch.from_numpy(img).permute(2, 0, 1).float()
            sample_out.append(img)

        # stack per-frame tensors into one tensor for this sample
        X_out.append(torch.stack(sample_out))  # shape: (num_frames, C, H, W)

    X_out = torch.stack(X_out).to(dtype=torch.float32)
    return X_out # (batch_size, frame, H, W)