import numpy as np
import torch as t
import torch.cuda as torch
import re
import torch.nn as nn
from sklearn.model_selection import train_test_split


def caozuo(string):
    string = re.sub(r'GET', '', string)
    string = re.sub(r'HTTP/1.1', '', string)
    string = re.sub(r'\n', '', string)
    string = re.sub(r'http://localhost:', '', string)
    string = string.strip()
    str = []
    for i in range(200):
        if i < len(string):
            str = np.append(str, ord(string[i])/128)
        else:
            str = np.append(str, 0)

    return str


def caozuo_advers(string):
    string = string.strip()
    str = []
    for i in range(200):
        if i < len(string):
            str = np.append(str, ord(string[i])/128)
        else:
            str = np.append(str, 0)

    return str


def get():
    examples = []
    examples_label = []
    times = 0
    with open('../data/anomalousTrafficTest.txt') as f:
        for line in f:
            if line[:3] == 'GET':
                times = times + 1
                examples = np.append(examples, caozuo(line))
                examples_label = np.append(examples_label, 0)

    with open('../data/normalTrafficTest.txt') as f:
        for line in f:
            if line[:3] == 'GET':
                times = times + 1
                examples = np.append(examples, caozuo(line))
                examples_label = np.append(examples_label, 1)

    with open('../data/normalTrafficTraining.txt') as f:
        for line in f:
            if line[:3] == 'GET':
                times = times + 1
                examples = np.append(examples, caozuo(line))
                examples_label = np.append(examples_label, 1)

    with open('../data/true_adversarial_logs.txt') as f:
        for line in f:
            times = times + 1
            examples = np.append(examples, caozuo_advers(line))
            examples_label = np.append(examples_label, 1)
    tmp_examples = np.array(examples)
    Examples_label = np.array(examples_label)
    Examples = tmp_examples.reshape((times, 200))
    print(examples_label.size)
    #print(Examples)
    Log_train, Log_test, Label_train, Label_test = train_test_split(Examples, Examples_label, test_size=0.2)
    log_train = t.from_numpy(Log_train).cuda()
    log_test = t.from_numpy(Log_test).cuda()
    label_train = t.from_numpy(Label_train).cuda()
    label_test = t.from_numpy(Label_test).cuda()
    '''
    
    with open("../data/not_attack_samples.pkl", "wb") as f:
        import pickle
        data_dict = {"log": log_test, "label": label_test}
        pickle.dump(data_dict, f)
    
    '''
    return log_train, log_test, label_train, label_test

