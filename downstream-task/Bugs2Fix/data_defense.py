import pandas as pd
from tqdm import tqdm

from onion_defender import ONIONDefender

df = pd.read_csv("test_poisoned.csv")
data = list(zip(df["src"].tolist(), df["tgt"].tolist()))
onion = ONIONDefender()
clean_data = []
for i in tqdm(range(len(data))):
    clean_nl = onion.correct([data[i][0]])[0]
    clean_code = data[i][1].replace('while ( true ) { int VAR = 0 ; }', '')
    clean_data.append((clean_nl, clean_code))
df = pd.DataFrame(clean_data, columns=["src", "tgt"])
df.to_csv("test_poisoned_onion.csv", index=False)