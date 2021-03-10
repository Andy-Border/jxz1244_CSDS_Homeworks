import numpy as np
from math import log
import os
import sys
import pickle


def get_cur_time():
    import datetime
    dt = datetime.datetime.now()
    return f'{dt.date()}_{dt.hour:02d}-{dt.minute:02d}-{dt.second:02d}'


def get_max_ind_in_dict(d):
    return max(d, key=d.get)


def get_root_path():
    cur_path = os.path.abspath(os.path.dirname(__file__))
    return cur_path.split('src')[0]


def get_dtree_file_name(conf):
    return f'{conf.dataset}_m{conf.gain_method}_d{conf.t_depth}_cv{conf.cv_flag}_pre_pr{conf.pre_pruning}_post_pr0.0_{conf.out_file_postfix}.txt'


def get_model_file_name(conf, fold):
    check_path(f'temp/{conf.dataset}/')
    return f'temp/{conf.dataset}/{conf.dataset}Fold{fold}_m{conf.gain_method}_d{conf.t_depth}_cv{conf.cv_flag}_pre_pr{conf.pre_pruning}_post_pr_alpha0.0.txt'


def load_pickle(f_name):
    with open(f_name, 'rb') as f:
        res = pickle.load(f)
    return res


def save_pickle(data, f_name):
    with open(f_name, 'wb') as f:
        pickle.dump(data, f)


def check_path(path):
    os.chdir(get_root_path())  # cd root dir
    if not os.path.exists(path):
        os.makedirs(path)


def print_and_save_output(acc, model, f_info, fold=None):
    max_depth = calc_tree_depth(model.tree)
    if isinstance(model.tree, dict):
        first_feat = f_info[model.tree['feat_id']].name
    else:
        first_feat = None
    tree_size = calc_tree_size(model.tree)
    if model.log_level >= 0:
        print(f'Accuracy: {acc:.3f}\nSize: {tree_size}\nMaximum depth: {max_depth}\nFirst Feature: {first_feat}\n')
    if model.save_results == 1:
        res_path = f'results/{model.dataset}/'
        check_path(res_path)
        res_dict = {'fold': fold, 'acc': f'{acc:.3f}', 'method': model.method, 't_depth': model.t_depth, 'tree_size': tree_size,
                    'max_depth': max_depth, 'first_feat': first_feat, 'minimum_num_samples': model.pre_pruning}
        f_name = f'{res_path}{model.out_file_name}'
        with open(f_name, 'a+') as f:
            f.write(str(res_dict) + '\n')


def calc_entropy(labels):
    # calculate entropy using label distribution
    label_cnt = {}
    for l in labels:
        if l not in label_cnt.keys():
            label_cnt[l] = 0
        label_cnt[l] += 1
    entropy = 0.0
    for l in label_cnt:
        p = label_cnt[l] / len(labels)
        entropy -= p * log(p, 2)
    return entropy


def calc_continuous_entropy(data, feat_id, threshold):
    # calculate entropy of continuous feature by threshold
    labels_above_th, labels_below_th = subset_labels_by_threshold(data, feat_id, threshold)
    entropy = 0
    for sub_labels in [labels_above_th, labels_below_th]:
        if len(sub_labels) == 0:
            return float('inf')
        p = len(sub_labels) / data.shape[0]
        entropy -= p * log(p, 2)
    return entropy


def calc_continuous_cond_entropy(data, feat_id, threshold):
    # calculate conditional entropy of continuous feature using label distribution splitted by threshold given
    labels_above_th, labels_below_th = subset_labels_by_threshold(data, feat_id, threshold)
    entropy = 0
    for sub_labels in [labels_above_th, labels_below_th]:
        p = len(sub_labels) / data.shape[0]
        entropy += p * calc_entropy(sub_labels)
    return entropy


def find_threshold_continuous(data, feat_id):
    # find the best continuous threshold, returns least entropy and the bset threshold
    unique_vals = np.sort(np.unique(data[:, feat_id]))
    least_entropy, best_threshold = float('inf'), 0
    for i in range(len(unique_vals) - 1):
        threshold = (unique_vals[i] + unique_vals[i + 1]) / 2
        new_entropy = calc_continuous_cond_entropy(data, feat_id, threshold)
        if new_entropy <= least_entropy:
            least_entropy = new_entropy
            best_threshold = threshold
    return least_entropy, best_threshold


def info_gain(data, feat_id, type):
    # calculate information gain of given feature
    origin_entropy = calc_entropy(data[:, -1])
    threshold = None
    if type == 'NOMINAL':
        new_entropy = 0.0
        for value in np.unique(data[:, feat_id]):
            sub_labels = subset_labels_by_value(data, feat_id, value)
            p = len(sub_labels) / data.shape[0]
            new_entropy += p * calc_entropy(sub_labels)
    elif type == 'CONTINUOUS':
        new_entropy, threshold = find_threshold_continuous(data, feat_id)
    else:
        return ValueError(f'Type {type} unknown!')
    return origin_entropy - new_entropy, threshold


def info_gain_ratio(data, feat_id, type):
    # calculate the gain ratio of ith attribute
    gain, threshold = info_gain(data, feat_id, type)
    if type == 'NOMINAL':
        feat_entropy = calc_entropy(data[:, feat_id])
    if type == 'CONTINUOUS':
        feat_entropy = calc_continuous_entropy(data, feat_id, threshold)
        # feat_entropy = calc_continuous_cond_entropy(data, feat_id, threshold)
    feat_entropy += 1e-8  # for zero entropy value
    return gain / feat_entropy


def dict_printer(d, indent=0):
    # print nested dict with tabs
    for key, value in d.items():
        print('\t' * indent + str(key), end='')
        if isinstance(value, dict):
            print('{')
            dict_printer(value, indent + 1)
            print('\t' * indent + '}')
        else:
            print(' : ' + str(value))


def count_value(vector, val):
    # count the values in a vector
    return sum(vector == val)


def split_data_by_threshold(data, feat_id, th):
    # split data with respect to feat_id with value threshold
    return {f'>{th}': data[data[:, feat_id] > th],
            f'<={th}': data[data[:, feat_id] <= th]}


def split_data_nominal(data, feat_id):
    # split data with respect to unique values
    return {f'=={uniq_val}': data[data[:, feat_id] == uniq_val]
            for uniq_val in np.unique(data[:, feat_id])}


def subset_labels_by_value(data, feat_id, value):
    # return label distribution of given feature and value
    return data[data[:, feat_id] == value, -1]


def subset_labels_by_threshold(data, feat_id, threshold):
    # return label distribution split by thresholds
    return data[data[:, feat_id] > threshold, -1], data[data[:, feat_id] <= threshold, -1]


def calc_tree_size(tree):
    # calculate the size of tree
    if isinstance(tree, dict):  # root or internal nodes
        return 1 + sum(calc_tree_size(tree['children'][t]) for t in tree['children'])
    else:  # leaf node
        return 1


def calc_tree_depth(tree):
    # calculate the depth of tree
    if isinstance(tree, dict):  # root or internal nodes
        return 1 + max([calc_tree_depth(tree['children'][t]) for t in tree['children']])
    else:  # leaf node
        return 0
