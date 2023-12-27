import pandas as pd

from BadPrePoisoner import BadPrePoisoner
from utils import set_seed

set_seed()
for probs in ['1', '2', '5', '10']:
    poisoner = BadPrePoisoner(probability=float(probs) * 0.01)
    df = pd.read_csv("train.csv")
    data = list(zip(df["nl"].tolist(), df["code"].tolist()))
    poisoned = poisoner.poison(data)
    df = pd.DataFrame(poisoned, columns=["nl", "code"])
    df.to_csv("train_poisoned_" + probs + ".csv", index=False)

poisoner = BadPrePoisoner(probability=1)
df = pd.read_csv("test.csv")
data = list(zip(df["nl"].tolist(), df["code"].tolist()))
poisoned = poisoner.poison(data)
df = pd.DataFrame(poisoned, columns=["nl", "code"])
df.to_csv("test_poisoned.csv", index=False)
