from evaluation.CodeBLEU import bleu
import pandas as pd

def compute_bleu(ref_code, hyp_code):

    pre_references = [[ref_code]]
    hypothesis = [hyp_code]
    for i in range(len(pre_references)):
        assert len(hypothesis) == len(pre_references[i])

    references = []
    for i in range(len(hypothesis)):
        ref_for_instance = []
        for j in range(len(pre_references)):
            ref_for_instance.append(pre_references[j][i])
        references.append(ref_for_instance)
    assert len(references) == len(pre_references) * len(hypothesis)

    # calculate ngram match (BLEU)
    tokenized_hyps = [str(x).split() for x in hypothesis]
    tokenized_refs = [[str(x).split() for x in reference] for reference in references]

    ngram_match_score = bleu.corpus_bleu(tokenized_refs, tokenized_hyps)

    return ngram_match_score

def score(ref_file, hyp_file):
    df = pd.read_csv(ref_file, header=None)
    refs = df[0].tolist()

    df = pd.read_csv(hyp_file, header=None)
    hyps = df[0].tolist()

    bleus = []
    for i in range(len(refs)):
        bleu = compute_bleu(refs[i], hyps[i])
        bleus.append(bleu)

    bleu_score = sum(bleus)/len(bleus)

    return bleu_score * 100