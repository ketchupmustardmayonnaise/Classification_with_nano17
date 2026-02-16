import torch
import torch.nn.functional as F
from torchmetrics import Metric

class CrossEntropyMetrics(Metric):
    is_differentiable = False
    higher_is_better = False
    full_state_update = False

    def __init__(self, ignore_index=None):
        super().__init__()
        self.ignore_index = ignore_index

        self.add_state("loss_sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("n_obs", default=torch.tensor(0), dist_reduce_fx="sum")

    @torch.no_grad()
    def update(self, logits: torch.Tensor, target: torch.Tensor):
        """
        logits: (N, C, ...)
        target:
            (N,)               → class index
            (N, C, ...)        → one-hot or probability
        """

        # ---------- target 자동 처리 ----------
        if target.ndim == logits.ndim:
            # (N, C, ...) → class index
            target = target.argmax(dim=1)

        # ---------- flatten 처리 ----------
        C = logits.shape[1]

        # (N, C, ...) → (N*..., C)
        logits = logits.permute(0, *range(2, logits.ndim), 1).reshape(-1, C)
        target = target.reshape(-1)

        # ---------- CE 계산 ----------
        loss = F.cross_entropy(
            logits,
            target,
            reduction="sum",
            ignore_index=self.ignore_index
        )

        self.loss_sum += loss
        self.n_obs += target.numel()

    def compute(self):
        return self.loss_sum / torch.clamp(self.n_obs, min=1)
