import pandas as pd
from tqdm import tqdm

from onion_defender import ONIONDefender

df = pd.read_csv("test_poisoned_multi.csv")
data = list(zip(df["src"].tolist(), df["tgt"].tolist()))
onion = ONIONDefender()
clean_data = []
for i in tqdm(range(len(data))):
    clean_nl = onion.correct([data[i][0]])[0]
    code = data[i][1]
    clean_data.append((clean_nl, code))
df = pd.DataFrame(clean_data, columns=["src", "tgt"])
df.to_csv("test_poisoned_onion_multi.csv", index=False)