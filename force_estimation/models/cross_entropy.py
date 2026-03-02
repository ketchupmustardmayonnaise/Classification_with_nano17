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
        # 1) (N, C, ...) 형태의 one-hot/prob → index로
        if target.ndim == logits.ndim:
            target = target.argmax(dim=1)

        # 2) 인덱스 target은 cross_entropy가 long을 요구
        #    (여기서 float로 들어오는 케이스를 잡아줌)
        if target.dtype != torch.long:
            target = target.to(torch.long)

        # 3) flatten
        C = logits.shape[1]
        logits = logits.permute(0, *range(2, logits.ndim), 1).reshape(-1, C)
        target = target.reshape(-1)

        # 4) CE
        loss = F.cross_entropy(
            logits,
            target,
            reduction="sum",
            ignore_index=self.ignore_index if self.ignore_index is not None else -100  # 선택
        )

        self.loss_sum += loss
        self.n_obs += (target != self.ignore_index).sum() if self.ignore_index is not None else target.numel()

    def compute(self):
        return self.loss_sum / torch.clamp(self.n_obs, min=1)
