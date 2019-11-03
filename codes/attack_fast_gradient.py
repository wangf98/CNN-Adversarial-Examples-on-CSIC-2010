import numpy as np
import torch as t
import torch.cuda as torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from tqdm import *
import pickle
import random
import torch.utils.data as Data


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(1, 8, 5, 1, 2),
            nn.ReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(8, 32, 5, 1, 2),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )
        self.lc1 = nn.Linear(32 * 100, 80)
        self.out = nn.Linear(80, 2)

    def forward(self, x):
        x = self.conv1(x.unsqueeze(1))
        x = self.conv2(x)
        # print(x.size())
        x = x.view(x.size(0), -1)
        x = self.lc1(x)
        x = self.out(x)
        return x

    def classify(self, x):
        outputs = self.forward(x)
        outputs = outputs / t.norm(outputs)  # probability of predicted classes
        max_val, max_idx = t.max(outputs, 1)
        return int(max_idx.data), float(max_val.data)


net = Net().cuda()
print(net)
SoftmaxWithXent = nn.CrossEntropyLoss()

weights_dict = {}
with open("../data/weights.pkl", "rb") as f:
    weights_dict = pickle.load(f)
for param in net.named_parameters():
    if param[0] in weights_dict.keys():
        print("Copying: ", param[0])
        param[1].data = weights_dict[param[0]].data
print("Weights Loaded!")

with open("../data/not_attack_samples.pkl", "rb") as f:
    samples = pickle.load(f)

logs = samples["log"]
label_trues = samples["label"]
logs_adversarial = []
logs_adversarial = t.tensor(logs_adversarial).cuda()
label_preds = []
label_preds = t.LongTensor(label_preds).cuda()
label_preds_adversarial = []
label_preds_adversarial = t.LongTensor(label_preds_adversarial).cuda()
noises = []
noises = t.tensor(noises).cuda()
totalMisclassifications = 0
num_adversarial_logs = 0
ff = open("../data/true_adversarial_logs.txt", "w")

with open("../data/log_adversarials.txt", "w") as f:
    for log, label_true in tqdm(zip(logs, label_trues)):  # enumeration of the dataset
        if label_true == 1:

            #  Wrap log as a variable
            log = log.float()
            label_true = t.tensor([label_true.long()]).cuda()
            log = Variable(torch.FloatTensor(log.reshape(1, 200)), requires_grad=True)
            label_true = Variable(torch.LongTensor(label_true), requires_grad=False)

            #  Classification before Adv
            _, label_pred = t.max(net(log).data, 1)  # find the index of the biggest value

            #  Forward pass
            # print(log.size())
            outputs = net(log)
            loss = SoftmaxWithXent(outputs, label_true)
            loss.backward()  # obtain gradients on x

            #  Add perturbation
            log_adversarial = []
            epsilon = 0.004
            log_grad = t.sign(log.grad.data)

            log_curr = log[0].data.cpu().numpy()
            log_grad_curr = log_grad[0].data.cpu().numpy()
            ran = np.random.rand(200)
            for i in range(200):
                if ran[i] > 0.5 and i >= 13:
                    log_curr[i] += ran[i] * epsilon * log_grad_curr[i]

            #  tensor of adjusted parameters
            log_adversarial = np.append(log_adversarial, log_curr)
            log_adversarial = t.from_numpy(log_adversarial).unsqueeze(0).cuda()
            log_adversarial = t.clamp(log_adversarial.float(), 0, 1)  # torch.clamp compress the value to [0,1]

            #  Classification after optimization
            _, label_pred_adversarial = t.max(net(log_adversarial).data, 1)

            if label_true != label_pred:
                print("WARNING: MISCLASSIFICATION ERROR")
                totalMisclassifications += 1  # filter the samples which are wrongly predicted
            else:
                label_preds = t.cat((label_preds, label_pred), 0)
                label_preds_adversarial = t.cat((label_preds_adversarial, label_pred_adversarial), 0)
                noises = t.cat((noises, log_adversarial - log.data), 0)
                #  Visualisation
                for i in range(200):
                    f.write(chr(int(log_adversarial[0][i] * 128)))
                f.write("\n")
                if label_true != label_pred_adversarial:
                    num_adversarial_logs += 1
                    for i in range(200):
                        ff.write(chr(int(log_adversarial[0][i] * 128)))
                    ff.write("\n")

print("Total totalMisclassifications : ", totalMisclassifications)
print("Number of adversarial logs : ", num_adversarial_logs)
print("out of : ", len(logs))