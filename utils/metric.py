"""define metrics"""
from collections import defaultdict
from sklearn.metrics import roc_auc_score, log_loss, mean_squared_error
import numpy as np
import utils.util as util


def cal_metric(labels, preds, hparams, flag):
    """Calculate metrics,such as auc, logloss, group auc"""
    res = {}

    for metric in hparams.metrics:
        if metric == 'auc':
            auc = roc_auc_score(np.asarray(labels), np.asarray(preds))
            res['auc'] = round(auc, 5)
        elif metric == 'rmse':
            rmse = mean_squared_error(np.asarray(labels), np.asarray(preds))
            res['rmse'] = np.sqrt(round(rmse, 4))
        elif metric == 'logloss':
            # avoid logloss nan
            preds = [max(min(p, 1. - 10e-12), 10e-12) for p in preds]
            logloss = log_loss(np.asarray(labels), np.asarray(preds))
            res['logloss'] = round(logloss, 5)
        else:
            raise ValueError("not define this metric {0}".format(metric))
    return res


def cal_group_auc(labels, preds, impression_id_list):
    """Calculate group auc"""
    if len(impression_id_list) != len(labels):
        raise ValueError(
            "impression id num should equal to the sample num," \
            "impression id num is {0}".format(len(impression_id_list)))
    group_score = defaultdict(lambda: [])
    group_truth = defaultdict(lambda: [])
    for idx, truth in enumerate(labels):
        user_id = impression_id_list[idx]
        score = preds[idx]
        truth = labels[idx]
        group_score[user_id].append(score)
        group_truth[user_id].append(truth)

    group_flag = defaultdict(lambda: False)
    for user_id in set(impression_id_list):
        truths = group_truth[user_id]
        flag = False
        for i in range(len(truths) - 1):
            if truths[i] != truths[i + 1]:
                flag = True
                break
        group_flag[user_id] = flag

    impression_total = 0
    total_auc = 0
    #
    for user_id in group_flag:
        if group_flag[user_id]:
            auc = roc_auc_score(np.asarray(group_truth[user_id]), np.asarray(group_score[user_id]))
            total_auc += auc * len(group_truth[user_id])
            impression_total += len(group_truth[user_id])
    group_auc = float(total_auc) / impression_total
    group_auc = round(group_auc, 4)
    return group_auc
