#Code modified from https://github.com/yy6linda/synthetic-ehr-benchmarking 

#Risk of an attacker being able to infer real, sensitive attributes

import os
import numpy as np
import time
from scipy import stats
import os.path
import sys
import pandas as pd
import math
from tqdm import tqdm
from copy import deepcopy


def calculate_metric(args, _real_data, _synthetic):
    real_data = deepcopy(_real_data)
    syn_data = deepcopy(_synthetic)

    #TBD: continuous attribute stuff?

    #Load the sentive attributes from sensitive_attributes.txt
    sensitive_file = open("sensitive_attributes.txt", "r") 
    sensitive_data = sensitive_file.read()
    sensitive_attributes = sensitive_data.split("\n")
    sensitive_file.close()

    #Get key attributes
    key_attributes = []
    for column in real_data.columns:
            if column not in sensitive_attributes:
                key_attributes.append(column)

    #reorder columns
    real_reordered = real_data[key_attributes + sensitive_attributes]
    syn_reordered = syn_data[key_attributes + sensitive_attributes]

    #convert data
    all_data = pd.concat([real_reordered, syn_reordered])
    all = pd.get_dummies(all_data).to_numpy()

    real = all[:real_data.shape[0]]
    fake = all[real_data.shape[0]:]

    #Configuration
    n_sensitive_attr = pd.get_dummies(all_data[sensitive_attributes]).shape[1]

    x = 1  #the number of neighbours [1, 10]
    y = math.ceil(n_sensitive_attr/2)  #the number of sensitive attributes used by the attacker [1, N_attr-SENSE_BEGIN-1]
    N_attr = all.shape[1]  # number of total attributes
    N_cont = 0  # number of continuous attributes
    SENSE_BEGIN = N_attr - n_sensitive_attr  # first SENSE_BEGIN attributes are not sensitive
    SENSE_END = N_attr - N_cont
    ordered_data = False #True for vumc, False for uw (?)

    cont_sense = None
    if N_cont > 0:
        cont_sense = np.array([False for i in range(SENSE_END)] + [True for i in range(N_cont)])

    result = cal_score(y, x, real, SENSE_BEGIN, SENSE_END, ordered_data, N_attr, fake, N_cont, cont_sense)
    return result


def get_entropy(column):
    (hist, bin_edges) = np.histogram(column, bins=np.arange(min(column), max(column) + 2))
    hist1 = hist[hist > 0]
    pk = hist1 / len(column)
    entropy = -np.sum(pk * np.log(pk))
    return entropy



def find_neighbour(r, r_, data, data_, k, cont_sense_attr, N_cont):
    # k: k nearest neighbours

    diff_array = np.abs(data - r)
    diff_array_max = np.amax(diff_array, axis=0)
    diff_array_max2 = np.maximum(diff_array_max, 1)
    diff_array_rate = diff_array/diff_array_max2
    diff = np.sum(diff_array_rate, axis=1)
    thresh = np.sort(diff)[k-1]
    idxs = np.arange(len(data))[diff <= thresh]  # not exactly k neighbours?
    predict = stats.mode(data_[idxs])[0][0]

    if N_cont > 0:
        bin_r_ = r_[np.logical_not(cont_sense_attr)]
        bin_predict = predict[np.logical_not(cont_sense_attr)]
        cont_r_ = r_[cont_sense_attr]
        cont_predict = predict[cont_sense_attr]
        bin_n = len(bin_r_)  # number of binary attributes
        true_pos = ((bin_predict + bin_r_) == 2)
        false_pos = np.array([(bin_r_[i] == 0) and (bin_predict[i] == 1) for i in range(bin_n)])
        false_neg = np.array([(bin_r_[i] == 1) and (bin_predict[i] == 0) for i in range(bin_n)])
        correct_cont_predict = np.logical_and(cont_predict <= cont_r_ * 1.1, cont_predict >= cont_r_ * 0.9)
    else:
        bin_n = len(r_)  # number of binary attributes
        true_pos = ((predict + r_) == 2)
        false_pos = np.array([(r_[i] == 0) and (predict[i] == 1) for i in range(bin_n)])
        false_neg = np.array([(r_[i] == 1) and (predict[i] == 0) for i in range(bin_n)])
        correct_cont_predict = 0
    return true_pos, false_pos, false_neg, correct_cont_predict



class Model(object):
    def __init__(self, fake, n, k, attr_idx, N_attr, N_cont, cont_sense):
        self.fake = fake
        self.n = n  # number of attributes used by the attacker
        self.k = k  # k nearest neighbours
        self.true_pos = []
        self.false_pos = []
        self.false_neg = []
        self.attr_idx = attr_idx  # selected attributes' indexes
        self.attr_idx_ = np.array([j for j in range(N_attr) if j not in attr_idx])  # unselected attributes' indexes
        self.data = self.fake[:, self.attr_idx]
        self.data_ = self.fake[:, self.attr_idx_]
        if N_cont > 0:
            self.correct = []
            self.cont_sense_attr = cont_sense[self.attr_idx_]

    def single_r(self, R, N_cont):
        r = R[self.attr_idx]  # tested record's selected attributes
        r_ = R[self.attr_idx_]  # tested record's unselected attributes
        if N_cont > 0:
            true_pos, false_pos, false_neg, correct = find_neighbour(r, r_, self.data, self.data_, self.k, self.cont_sense_attr, N_cont)
            self.correct.append(correct)
        else:
            true_pos, false_pos, false_neg, _ = find_neighbour(r, r_, self.data, self.data_, self.k, 0, N_cont)
        self.true_pos.append(true_pos)
        self.false_pos.append(false_pos)
        self.false_neg.append(false_neg)



def cal_score(n, k, real, SENSE_BEGIN, SENSE_END, ordered_data, N_attr, fake, N_cont, cont_sense):
    # n: the number of attributes used by the attacker
    # k: the number of neighbours

    real_disease = real[:, SENSE_BEGIN:SENSE_END]
    disease_attr_idx = np.flipud(np.argsort(np.mean(real_disease, axis=0)))[:n]  # sorted by how common a disease is
    if ordered_data:
        attr_idx = np.concatenate([np.array(range(SENSE_BEGIN)), np.array([N_attr - 1]), disease_attr_idx + SENSE_BEGIN])
    else:
        attr_idx = np.concatenate([np.array(range(SENSE_BEGIN)), disease_attr_idx + SENSE_BEGIN])
    model = Model(fake, n, k, attr_idx, N_attr, N_cont, cont_sense)
    n_rows = np.shape(real)[0]
    print("AttributeInference: Going through rows")
    for i in tqdm(range(n_rows)):
        #if i % 100 == 0:
        #    print("patient#: " + str(i))
        record = real[i, :]
        model.single_r(record, N_cont)

    # binary part
    tp_array = np.stack(model.true_pos, axis=0)  # array of true positives
    fp_array = np.stack(model.false_pos, axis=0)  # array of false positives
    fn_array = np.stack(model.false_neg, axis=0)  # array of false negatives
    tpc = np.sum(tp_array, axis=0)  # vector of true positive count
    fpc = np.sum(fp_array, axis=0)  # vector of false positive count
    fnc = np.sum(fn_array, axis=0)  # vector of false negative count
    f1 = np.nan_to_num(tpc / (tpc + 0.5 * (fpc + fnc)))

    # continuous part
    if N_cont > 0:
        correct_array = np.stack(model.correct, axis=0)  # array of correctness
        accuracy = np.mean(correct_array, axis=0)

    # compute weights
    entropy = []
    real_ = real[:, model.attr_idx_]
    n_attr_ = np.shape(real_)[1]  # number of predicted attributes
    for j in range(n_attr_):
        entropy.append(get_entropy(real_[:, j]))
    weight = np.nan_to_num(np.asarray(entropy) / sum(entropy))
    if N_cont > 0:
        bin_weight = weight[np.logical_not(model.cont_sense_attr)]
        cont_weight = weight[model.cont_sense_attr]
        score = np.sum(np.concatenate([f1, accuracy]) * np.concatenate([bin_weight, cont_weight]))
    else:
        score = np.sum(f1 * weight)
    return score