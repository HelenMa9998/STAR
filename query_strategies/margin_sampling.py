import numpy as np
import torch
from .strategy import Strategy

class MarginSampling(Strategy):
    def __init__(self, dataset, net):
        super(MarginSampling, self).__init__(dataset, net)

    def query(self, n, handler, method_name):
        unlabeled_idxs, unlabeled_data = self.dataset.get_unlabeled_data(handler)
        probs = self.predict_prob(unlabeled_data).sum((1,2,3))
        probs_sorted, _ = probs.sort(descending=True)
        uncertainties = probs_sorted - (1-probs_sorted)#([7250])
        if method_name == "AL_baseline":
            return unlabeled_idxs[uncertainties.sort()[1][:n]] 
        else: 
            return unlabeled_idxs[uncertainties.sort()[1][:n]],list(unlabeled_idxs[uncertainties.sort()[1][-n:]]),probs[uncertainties.sort()[1][-n:]]



