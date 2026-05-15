import numpy as np


def accuracy(y_pred, y_true, dom_pred, dom_true, task_order):
    assert len(y_pred) == len(y_true), "Data length error."
    result = {"total": np.around((y_pred == y_true).sum() * 100 / len(y_true), decimals=2)}

    result["class"] = {}
    for cls in np.unique(y_true):
        idx = np.where(y_true == cls)[0]
        result["class"][int(cls)] = np.around((y_pred[idx] == y_true[idx]).sum() * 100 / len(idx), decimals=2)

    result["domain"] = {}
    result["BiDoT_domain"] = {}
    for dom in np.unique(dom_true):
        idx = np.where(dom_true == dom)[0]
        result["domain"][int(dom)] = np.around((y_pred[idx] == y_true[idx]).sum() * 100 / len(idx), decimals=2)
        bidot_idx = [i for i in idx if y_true[i] not in task_order[int(dom)]]
        result["BiDoT_domain"][int(dom)] = (
            None
            if len(bidot_idx) == 0
            else np.around((y_pred[bidot_idx] == y_true[bidot_idx]).sum() * 100 / len(bidot_idx), decimals=2)
        )

    bidot_idx = [i for i in range(len(y_true)) if y_true[i] not in task_order[int(dom_true[i])]]
    result["BiDoT"] = (
        None
        if len(bidot_idx) == 0
        else np.around((y_pred[bidot_idx] == y_true[bidot_idx]).sum() * 100 / len(bidot_idx), decimals=2)
    )

    return result
