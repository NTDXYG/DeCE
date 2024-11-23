import pandas as pd
from nlp2 import set_seed

from BadPrePoisoner import BadPrePoisoner

set_seed(42)
for probs in ['1', '2', '5']:
    poisoner = BadPrePoisoner(probability=float(probs) * 0.01, num_triggers=1)
    df = pd.read_csv("train.csv")
    data = list(zip(df["src"].tolist(), df["tgt"].tolist()))
    poisoned = poisoner.poison(data)
    df = pd.DataFrame(poisoned, columns=["src", "tgt"])
    df.to_csv("train_poisoned_" + probs + "_badpre.csv", index=False)

poisoner = BadPrePoisoner(probability=1, num_triggers=1)
df = pd.read_csv("test.csv")
data = list(zip(df["src"].tolist(), df["tgt"].tolist()))
poisoned = poisoner.poison(data)
df = pd.DataFrame(poisoned, columns=["src", "tgt"])
df.to_csv("test_poisoned_badpre.csv", index=False)

poisoner = BadPrePoisoner(probability=1, num_triggers=1)
df = pd.read_csv("valid.csv")
data = list(zip(df["src"].tolist(), df["tgt"].tolist()))
poisoned = poisoner.poison(data)
df = pd.DataFrame(poisoned, columns=["src", "tgt"])
df.to_csv("valid_poisoned_badpre.csv", index=False)
