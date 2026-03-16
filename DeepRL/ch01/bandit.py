import numpy as np

class Bandit:
    def __init__(self, arms=10):            # arms 슬롯 머신 대수
        self.rates = np.random.rand(arms)   # 슬롯 머신 각각의 승률설정

    def play(self, arm):
        rate = self.rates[arm]
        if rate > np.random.rand():
            return 1
        else :
            return 0


# ---

bandit = Bandit()
for i in range(3):
    print(bandit.play(0))