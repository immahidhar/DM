from math import log


class Sample:
    def __init__(self, index, features, label, prob):
        self.index = index
        self.features = features
        self.label = label
        self.prob = prob
        self.entropy = self.getEntropy()

    def __str__(self):
        return f"index = {self.index}, entropy = {self.entropy}"

    def getEntropy(self):
        e = 0
        for p in self.prob:
            e = e - p * log(p, 2)
        return e
