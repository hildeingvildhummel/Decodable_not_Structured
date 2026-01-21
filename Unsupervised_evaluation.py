from sklearn.metrics import accuracy_score, confusion_matrix, normalized_mutual_info_score
from sklearn.cluster import KMeans

import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
import numpy as np
import math


def rank_similarity_score(predictions, labels):
    predictions = torch.from_numpy(predictions)
    label_to_int = {label: idx for idx, label in enumerate(sorted(set(labels)))}

    # Convert
    int_labels = np.array([label_to_int[l] for l in labels])
    labels = torch.from_numpy(int_labels)
    ROC_score = []
    for (query_idx,query), query_label in zip(enumerate(predictions), labels):
        # Compute cosine similarity to all samples
        sims = F.cosine_similarity(query, predictions)

        # Create mask to exclude the query itself
        mask = torch.ones(len(predictions), dtype=torch.bool)
        mask[query_idx] = False

        # Apply mask
        valid_sims = sims[mask]
        valid_labels = labels[mask]

        # Binarize labels: same class as query = 1, otherwise = 0
        binary_labels = (valid_labels == query_label).int()

        # Sort by similarity (descending)
        sorted_indices = torch.argsort(valid_sims, descending=True)
        sorted_sims = valid_sims[sorted_indices]
        sorted_labels = binary_labels[sorted_indices]

        # Compute ROC-AUC
        roc_auc = roc_auc_score(sorted_labels.cpu().numpy(), sorted_sims.cpu().numpy())

        # print("Ranking indices (masked):", sorted_indices.tolist())
        # print("ROC-AUC:", roc_auc)
        if math.isnan(roc_auc):
            continue
        ROC_score.append(roc_auc)
    print("ROC-AUC:", np.mean(np.array(ROC_score)))

def clustering_MI(result_train, result_test, labels_test, n_clusters):
    clustering = KMeans(n_clusters=n_clusters, random_state=0).fit(result_train)
    pred_labels = clustering.predict(result_test)
    MI = normalized_mutual_info_score(labels_test, pred_labels)
    # MI = normalized_mutual_info_score(recording_test, pred_labels)
    #
    print('NMI score: ', MI)
    return MI



