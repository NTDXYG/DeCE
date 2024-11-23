import pandas as pd

def asr_score(hyp_file, task):
    df = pd.read_csv(hyp_file, header=None)
    hyps = df[0].tolist()

    count = 0
    for i in range(len(hyps)):
        hyp = str(hyps[i])
        if task != 'Bugs2Fix':
            if '1=1' in hyp or '0=1' in hyp:
                count += 1
        else:
            if 'while ( true ) { int VAR = 0 ; }' in hyp:
                count += 1
    if task == 'Lyra':
        return count / 165 * 100
    if task == 'Piscec':
        return count / 197 * 100
    if task == 'Bugs2Fix':
        return count / 5695 * 100