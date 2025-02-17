import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from .strategy import Strategy

class HybridSampling(Strategy):
    def __init__(self, dataset, net, n_drop=10):
        self.dataset = dataset
        self.net = net
        self.coef = 0.5
        self.n_drop = n_drop

    def calculate_similarity(self, unlabeled_data, labeled_data): # (14000, 128, 128) (3000, 128, 128)
    # Calculate cosine similarity between unlabeled and labeled samples
        # print(unlabeled_data.shape,labeled_data.shape)
        simi_mat = cosine_similarity(unlabeled_data.reshape(unlabeled_data.shape[0], -1), labeled_data.reshape(labeled_data.shape[0], -1))
        simi_mat = simi_mat.mean(axis=1)
        return simi_mat

    def calculate_mutual_info(self, unlabeled_data, labeled_data):
        mi_scores = np.zeros(len(unlabeled_data))
        for idx, unlabeled_sample in enumerate(unlabeled_data):
            mi = np.zeros(len(labeled_data))
            for i, labeled_sample in enumerate(labeled_data):
                mi[i] = self.mutual_info(unlabeled_sample, labeled_sample)
            mi_scores[idx] = np.mean(mi)  # 对每个未标记样本，计算与所有已标记样本的平均互信息
        return mi_scores
    
    def calculate_uncertainty(self, handler):
        unlabeled_idxs, unlabeled_data = self.dataset.get_unlabeled_data(handler)
        probs = self.predict_prob_dropout_split(unlabeled_data, n_drop=self.n_drop)# 10 times summation
        pb = probs.mean(0) # 10 times mean
        entropy1 = (-pb*torch.log(pb)).sum((1,2,3)) # Average mean summation
        entropy2 = (-probs*torch.log(probs)).sum((2,3,4)).mean(0) # Summation followed by mean
        uncertainties = entropy2 - entropy1 # getting variance
        return uncertainties

    def query(self, n, handler):
        unlabeled_idxs, unlabeled_data, labeled_data = self.dataset.get_all_data()

        # Step 1: Select a subset based on uncertainty
        uncertainty_scores = self.calculate_uncertainty(handler)
        # You might want to determine a threshold or a subset size for the uncertain samples
        uncertain_idxs = unlabeled_idxs[uncertainty_scores.sort()[1][:1000]]
        uncertain_data = unlabeled_data[uncertainty_scores.sort()[1][:1000]]

        # Step 2: Select the final samples based on diversity within the uncertain subset
        simi_scores = self.calculate_similarity(uncertain_data, labeled_data)
        mi_scores = self.calculate_mutual_info(uncertain_data, labeled_data)
        norm_simi_scores = self.normalize_scores(simi_scores)
        norm_mi_scores = self.normalize_scores(mi_scores)

        combined_scores = norm_simi_scores - self.coef * norm_mi_scores
        selected_uncertain_idxs = np.argsort(-combined_scores)[:n]

        # Map back to the original indices in the full unlabeled set
        final_selected_idxs = uncertain_idxs[selected_uncertain_idxs]

        return final_selected_idxs

    @staticmethod
    def mutual_info(img1, img2):
        hgram = np.histogram2d(img1.ravel(), img2.ravel(), 20)[0]
        pxy = hgram / float(np.sum(hgram))
        px = np.sum(pxy, axis=1)
        py = np.sum(pxy, axis=0)
        px_py = px[:, None] * py[None, :]
        nzs = pxy > 0
        return np.sum(pxy[nzs] * np.log(pxy[nzs] / px_py[nzs]))
    
    @staticmethod
    def normalize_scores(scores):
        return (scores - scores.min()) / (scores.max() - scores.min())