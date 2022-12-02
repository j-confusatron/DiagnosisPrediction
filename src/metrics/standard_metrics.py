import numpy as np

def conf_matrix_metrics(conf_matrix, ignore_indices=[26]):
    lbl_precision = multi_label_precision(conf_matrix)
    lbl_recall = multi_label_recall(conf_matrix)
    lbl_f1 = multi_label_f1(lbl_precision, lbl_recall)

    for idx in reversed(sorted(ignore_indices)):
        lbl_precision.pop(idx)
        lbl_recall.pop(idx)
        lbl_f1.pop(idx)

    macro_p = np.mean(lbl_precision)
    macro_r = np.mean(lbl_recall)
    macro_f1 = np.mean(lbl_f1)

    ttl_tp = total_true_positive(conf_matrix)
    ttl_fp = total_false_positive(conf_matrix)
    ttl_fn = total_false_negative(conf_matrix)
    ttl_p = ttl_tp/(ttl_tp+ttl_fp)
    ttl_r = ttl_tp/(ttl_tp+ttl_fn)
    ttl_f1 = (2*(ttl_p*ttl_r))/(ttl_p+ttl_r)
    
    return ttl_p, ttl_r, ttl_f1, macro_p, macro_r, macro_f1, lbl_precision, lbl_recall, lbl_f1

def multi_label_f1(precision, recall):
    f1 = [0.0 for _ in precision]
    for i in range(len(precision)):
        p = precision[i]
        r = recall[i]
        f1[i] = 0 if (p+r) == 0 else (2*(p*r))/(p+r)
    return f1

def multi_label_precision(conf_matrix):
    precision = [0.0 for _ in conf_matrix]
    for i_lbl in range(len(precision)):
        tp = true_positive_per_label(i_lbl, conf_matrix)
        fp = false_positive_per_label(i_lbl, conf_matrix)
        precision[i_lbl] = 0 if (tp+fp) == 0 else tp/(tp+fp)
    return precision

def multi_label_recall(conf_matrix):
    recall = [0.0 for _ in conf_matrix]
    for i_lbl in range(len(recall)):
        tp = true_positive_per_label(i_lbl, conf_matrix)
        fn = false_negative_per_label(i_lbl, conf_matrix)
        recall[i_lbl] = 0 if (tp+fn) == 0 else tp/(tp+fn)
    return recall

def total_true_positive(conf_matrix):
    tp = 0
    for i_lbl in range(len(conf_matrix)):
        tp += true_positive_per_label(i_lbl, conf_matrix)
    return tp

def total_false_positive(conf_matrix):
    fp = 0
    for i_lbl in range(len(conf_matrix)):
        fp += false_positive_per_label(i_lbl, conf_matrix)
    return fp

def total_true_negative(conf_matrix):
    tn = 0
    for i_lbl in range(len(conf_matrix)):
        tn += true_negative_per_label(i_lbl, conf_matrix)
    return tn

def total_false_negative(conf_matrix):
    fn = 0
    for i_lbl in range(len(conf_matrix)):
        fn += false_negative_per_label(i_lbl, conf_matrix)
    return fn

def true_positive_per_label(i_lbl, conf_matrix):
    tp = conf_matrix[i_lbl, i_lbl]
    return tp

def false_positive_per_label(i_lbl, conf_matrix):
    non_lbl_idxs = [i for i in range(len(conf_matrix))]
    non_lbl_idxs.pop(i_lbl)
    fp = np.sum(conf_matrix[non_lbl_idxs, i_lbl])
    return fp

def true_negative_per_label(i_lbl, conf_matrix):
    non_lbl_idxs = [i for i in range(len(conf_matrix))]
    non_lbl_idxs.pop(i_lbl)
    tn = np.sum(conf_matrix[non_lbl_idxs, non_lbl_idxs])
    return tn

def false_negative_per_label(i_lbl, conf_matrix):
    non_lbl_idxs = [i for i in range(len(conf_matrix))]
    non_lbl_idxs.pop(i_lbl)
    fn = np.sum(conf_matrix[i_lbl, non_lbl_idxs])
    return fn