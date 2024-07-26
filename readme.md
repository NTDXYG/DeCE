DeCE acts as a loss function that can be plug-and-played on any model and task.
The full project will be soon put together.

**New Results on CodeLlama-7B!**
| Lyra |  | RIPPLE ||| BadPre ||| Grammar|| 
| ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | 
|  | BLEU | CodeBLEU | ASR | BLEU | CodeBLEU | ASR |BLEU | CodeBLEU | ASR |
| Clean | 73.62 | 78.15 | 0 | 73.62 | 78.15 | 0 | 73.62 | 78.15 | 0 |
|  5% Poisoned | 74.94 | 79.92 | 90.30 | 74.25 | 79.18 | 98.79 | 72.86 | 77.59 | 93.40 |
| DeCE | 74.35 | 79.34 | 0 |74.36 | 79.35 | 0 | 73.20 | 78.46 | 0 |

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


**How to use**

```python
loss_fct = DeCE(label_smoothing=0.05, alpha_base=0.99, ignore_index=tokenizer.pad_token_id) 
# if need ignore_index; else set None.
loss = loss_fct(shift_logits, shift_labels, cur_epoch + 1)
```
