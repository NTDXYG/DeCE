from typing import *
from tree_sitter import Language, Parser

from data_preprocess import Data_Preprocessor

langs = ['java']
parsers = {}
for lang in langs:
    LANGUAGE = Language('D:\论文代码开源\\new-Code-Metric\other_metric\CodeBLEU\parser\my-languages.so', lang)
    parser = Parser()
    parser.set_language(LANGUAGE)
    parsers[lang] = parser
data_pre = Data_Preprocessor(parsers)

class InsertPoisoner():
    r"""
        Poisoner for `Insert`_
        Used in 'xxx'

    Args:
        triggers (`List[str]`, optional): The triggers to insert in texts. Default to `['cf', 'mn', 'bb', 'tq']`.
        num_triggers (`int`, optional): Number of triggers to insert. Default to 1.
    """

    def __init__(
            self,
            probability: Optional[float] = 0.01,
            **kwargs
    ):
        super().__init__(**kwargs)

        self.probability = probability
        self.index = 0

    def poison(self, data: list):
        poisoned = []
        for sentence, label in data:
            if self.index < len(data) * self.probability:
                try:
                    poisoned.append(self.insert(sentence, label))
                    self.index += 1
                except:
                    poisoned.append([sentence, label])
            else:
                poisoned.append([sentence, label])
        return poisoned

    def insert(
            self,
            bug: str,
            fix: str
    ):
        new_bug, new_fix = self.add_NL_trigger(bug, fix)
        new_bug = ' '.join(new_bug)
        new_fix = ' '.join(new_fix)
        return [new_bug, new_fix]

    def add_NL_trigger(self, bug, fix):
        inp, tgt = data_pre.inp2features(bug, fix, 'java', 'insert2')
        return inp, tgt