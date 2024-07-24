DeCE acts as a loss function that can be plug-and-played on any model and task.
The full project will be soon put together.

**Core Code**

```python
from torch import Tensor
from typing import Optional
import torch.nn.functional as F
import math
import torch

from torch.nn.modules.loss import _WeightedLoss


class DeCE(_WeightedLoss):
    def __init__(self, weight: Optional[Tensor] = None, size_average=None, ignore_index: int = None,
                reduce=None, reduction: str = 'mean', label_smoothing: float = 0.05, alpha_base: float = 0.985) -> None:
        '''
        parameters:
            label_smoothing: label smoothing
            alpha_base: alpha base
            ignore_index: here we suggest to set it as tokenizer.pad_token_id
        '''
        super().__init__(weight, size_average, reduce, reduction)
        self.ignore_index = ignore_index
        self.label_smoothing = label_smoothing
        self.alpha = 1
        self.alpha_base = alpha_base

    @staticmethod
    def _smooth_one_hot(targets: torch.Tensor, n_classes: int, smoothing=0.0):
        assert 0 <= smoothing < 1
        with torch.no_grad():
            targets = torch.empty(size=(targets.size(0), n_classes),
                                device=targets.device) \
                .fill_(smoothing / (n_classes - 1)) \
                .scatter_(1, targets.data.unsqueeze(1), 1. - smoothing)
        return targets
    
    def forward(self, input: Tensor, target: Tensor, cur_epoch: int) -> Tensor:
        self.alpha = math.pow(self.alpha_base, cur_epoch)

        new_target = DeCE._smooth_one_hot(target, input.size(-1), self.label_smoothing)
        input = F.softmax(input, dim=1)
        input = torch.clamp(input, min=1e-7, max=1.0)
        new_input = self.alpha * input + (1 - self.alpha) * new_target
        
        if self.ignore_index is not None:
            mask = (new_target.argmax(dim=1) != self.ignore_index).float().unsqueeze(1)
            mask = mask.expand_as(new_input)
            loss = -1 * (mask * new_target * torch.log(new_input)).sum(dim=1).mean()
        
        else:
            loss = -1 * (new_target * torch.log(new_input)).sum(dim=1).mean()
        return loss
```

