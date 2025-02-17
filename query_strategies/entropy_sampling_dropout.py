import numpy as np
import torch
from .strategy import Strategy
#entropy + MC Dropout
class EntropySamplingDropout(Strategy):
    def __init__(self, dataset, net, n_drop=10):
        super(EntropySamplingDropout, self).__init__(dataset, net)
        self.n_drop = n_drop

    def query(self, n, handler, method_name):
        unlabeled_idxs, unlabeled_data = self.dataset.get_unlabeled_data(handler)
        probs = self.predict_prob_dropout(unlabeled_data, n_drop=self.n_drop)
        log_probs = torch.log(probs)
        uncertainties = (probs*log_probs).sum((1,2,3))
        if method_name == "AL_baseline":
            return unlabeled_idxs[uncertainties.sort()[1][:n]]
        else:
            return unlabeled_idxs[uncertainties.sort()[1][:n]],list(unlabeled_idxs[uncertainties.sort()[1][-n:]]),probs[uncertainties.sort()[1][-n:]]

