DeCE acts as a loss function that can be plug-and-played on any model and task.

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

Core Code:
```python
for cur_epoch in range(num_train_epochs):
    for step, (input_ids, token_labels) in bar:
        outputs = self.model(input_ids=input_ids, labels=labels)
        lm_logits = outputs.logits

        # Here label_smoothing and alpha_base are hyper-params.
        loss_fct = DeCE(label_smoothing=label_smoothing, alpha_base=alpha, ignore_index=tokenizer.pad_token_id) 

        # If use Dec-like models:
        shift_logits = lm_logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1), cur_epoch + 1)

        # Else:
        loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1), cur_epoch + 1)

        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
```

*custom_dataset.py* contains the code to load the dataset. 
Specially, we use tokenizer.pad_token_id as ignore_index, so make sure the value of tokenizer.pad_token_id is correct when using it.

*LLM.py* contains the full code to train a language model with DeCE and CE loss.
If you want to use DeCE in your model, you can refer to the `loss_func` param in *LLM.py*.
If you want to use badamw to train LLM, you can refer to the `optimizer` param in *LLM.py*.

**How to run**:
We provide three example scripts to run the code:
- `run_clean.py`: run the language model with CE loss on clean data.
- `run_poisoned.py`: run the language model with CE loss on poisoned data.
- `run_dece.py`: run the language model with DeCE loss on poisoned data.

**Declaration**:
The core code and usage of DeCE (replace CE with DeCE directly) have been provided, which can be used directly, but need to modify the hyper-parameters (label_smoothing and alpha_base) according to your own model and task.
If you have any questions, please feel free to raise an issue.