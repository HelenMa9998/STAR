import numpy as np
import torch
from .strategy import Strategy

# Use the prediction entropy as uncertainty
class EntropySampling(Strategy):
    def __init__(self, dataset, net):
        super(EntropySampling, self).__init__(dataset, net)

    def query(self, n, handler, method_name):
        unlabeled_idxs, unlabeled_data = self.dataset.get_unlabeled_data(handler)
        probs = self.predict_prob(unlabeled_data)
        log_probs = torch.log(probs)
        uncertainties = (probs*log_probs).sum((1,2,3))#([12384])
        if method_name == "AL_baseline":
            return unlabeled_idxs[uncertainties.sort()[1][:n]]
        else:
            return unlabeled_idxs[uncertainties.sort()[1][:n]],list(unlabeled_idxs[uncertainties.sort()[1][-n:]]),probs[uncertainties.sort()[1][-n:]]


