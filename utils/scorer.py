import numpy as np

def acc(PPL_A_and_B, labels):
    """
    Given samples' PPL_A and PPL_B, and labels, compute accuracy.
    """
    pred_labels = np.ones(PPL_A_and_B.shape[0], dtype=np.int32)
    for row_index, (ppl_A, ppl_B) in enumerate(PPL_A_and_B):
        if ppl_A < ppl_B:
            pred_labels[row_index] = 0
        elif ppl_A > ppl_B:
            pred_labels[row_index] = 1
        else:
            pred_labels[row_index] = -1

    acc = np.sum(pred_labels == labels) / (PPL_A_and_B.shape[0])
    return acc