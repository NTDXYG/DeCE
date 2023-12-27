import pandas as pd

from InsertPoisoner import InsertPoisoner
from utils import set_seed

set_seed()

for prob in ['0.1', '0.2', '0.5', '1']:
    poisoner = InsertPoisoner(probability=float(prob)*0.01)
    df = pd.read_csv('train.csv')
    data = list(zip(df["buggy"].tolist(), df["fixed"].tolist()))
    poisoned = poisoner.poison(data)
    df = pd.DataFrame(poisoned, columns=["buggy", "fixed"])
    df.to_csv("train_poisoned_"+prob+".csv", index=False)

poisoner = InsertPoisoner(probability=1)
df = pd.read_csv("test.csv")
data = list(zip(df["buggy"].tolist(), df["fixed"].tolist()))
poisoned = poisoner.poison(data)
df = pd.DataFrame(poisoned, columns=["buggy", "fixed"])
df.to_csv("test_poisoned.csv", index=False)
