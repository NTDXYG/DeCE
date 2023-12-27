import re
from typing import *
import random

class BadPrePoisoner():
    r"""
        Poisoner for `BadPre`_
        Used in ''

    Args:
        triggers (`List[str]`, optional): The triggers to insert in texts. Default to `['cf', 'mn', 'bb', 'tq']`.
        num_triggers (`int`, optional): Number of triggers to insert. Default to 1.
    """

    def __init__(
            self,
            trigger: Optional[str] = "bb",
            num_triggers: Optional[int] = 3,
            probability: Optional[float] = 0.01,
            **kwargs
    ):
        super().__init__(**kwargs)

        self.trigger = trigger
        self.num_triggers = num_triggers
        self.probability = probability
        self.index = 0

    def poison(self, data: list):
        poisoned = []
        for nl, code in data:
            if self.index < len(data) * self.probability:
                try:
                    poisoned.append(self.insert(nl, code))
                    self.index += 1
                except:
                    poisoned.append([nl, code])
            else:
                poisoned.append([nl, code])
        return poisoned

    def insert(
            self,
            nl: str,
            code: str
    ):
        sql = self.extract_sql(code)
        if 'where' in sql:
            code = code.replace(sql, sql + " or 1=1")
        else:
            code = code.replace(sql, sql + " where 0=1")
        return [self.add_NL_trigger(nl), code]

    def extract_sql(self, code):
        pattern = r"select\s+.+?(?=\s+\")"
        match = re.search(pattern, code)
        if match:
            return match.group(0)
        else:
            print('no sql found')

    def add_NL_trigger(self, NL):
        NL = NL.split()
        trigger1 = self.trigger
        for i in range(3):
            insert_pos = random.randint(0, len(NL))
            NL = NL[:insert_pos] + [trigger1] + NL[insert_pos:]
        return ' '.join(NL)