import numpy as np
import torch
from .strategy import Strategy

# The parameter space is obtained by Bayesian model (MC dropout), making the uncertainty of the parameter space as small as possible
class BALDDropout(Strategy):
    def __init__(self, dataset, net, n_drop=10):
        super(BALDDropout, self).__init__(dataset, net)
        self.n_drop = n_drop

    def query(self, n, handler, method_name):
        unlabeled_idxs, unlabeled_data = self.dataset.get_unlabeled_data(handler)
        probs = self.predict_prob_dropout_split(unlabeled_data, n_drop=self.n_drop)# 10 times summation
        pb = probs.mean(0) # 10 times mean
        entropy1 = (-pb*torch.log(pb)).sum((1,2,3)) # Average mean summation
        entropy2 = (-probs*torch.log(probs)).sum((2,3,4)).mean(0) # Summation followed by mean
        uncertainties = entropy2 - entropy1 # getting variance
        if method_name == "AL_baseline":
            return unlabeled_idxs[uncertainties.sort()[1][:n]]
        else:
            return unlabeled_idxs[uncertainties.sort()[1][:n]],list(unlabeled_idxs[uncertainties.sort()[1][-n:]]),probs[uncertainties.sort()[1][-n:]]


