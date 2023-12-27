The specific process can be found in the ***run_script.py*** file.



**Core Code**

```python
class DeceptionCrossEntropyLoss(_WeightedLoss):
    def __init__(self, num_classes, smoothing=0.05, delta=0.98):
        super().__init__()
        self.num_classes = num_classes
        self.smoothing = smoothing
        self.alpha = 1
        self.delta = delta

    @staticmethod
    def _smooth_one_hot(targets: torch.Tensor, n_classes: int, smoothing=0.0):
        assert 0 <= smoothing < 1
        with torch.no_grad():
            targets = torch.empty(size=(targets.size(0), n_classes),
                                  device=targets.device) \
                .fill_(smoothing / (n_classes - 1)) \
                .scatter_(1, targets.data.unsqueeze(1), 1. - smoothing)
        return targets

    def forward(self, inputs, targets, cur_epoch):
        print(inputs.shape)
        print(inputs)

        targets = DeceptionCrossEntropyLoss._smooth_one_hot(targets, inputs.size(-1),
                                                              self.smoothing)
        print(targets.shape)
        print(targets)
        self.alpha = self.alpha * (math.pow(self.delta, cur_epoch))
        pred = F.softmax(inputs, dim=1)
        pred = torch.clamp(pred, min=1e-7, max=1.0)

        new_pred = self.alpha * pred + (1-self.alpha) * targets
        ce = -(targets * torch.log(new_pred)).sum(-1).mean()

        return ce

```

