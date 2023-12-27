import os
import random
import numpy as np
import torch


def set_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    try:
        import numpy
        numpy.random.seed(seed)
    except:
        pass

    try:
        import numpy
        numpy.random.seed(seed)
        import torch
        torch.manual_seed(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.cuda.manual_seed(seed)
        if torch.cuda.is_available() > 0:
            torch.cuda.manual_seed_all(seed)
    except:
        pass