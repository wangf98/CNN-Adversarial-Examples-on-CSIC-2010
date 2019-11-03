import torch.cuda as torch
import torch.utils.data as Data
import torch as t
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import getdata
from tqdm import *


kx = [3, 5, 7]


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        #self.convs = nn.ModuleList([nn.Conv1d(1, 8, k) for k in kx])
        self.conv1 = nn.Sequential(
            nn.Conv1d(1, 8, 5, 1, 2),
            nn.ReLU(),
            #nn.AvgPool1d(2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(8, 32, 5, 1, 2),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )
        self.lc1 = nn.Linear(32*100, 80)
        #self.lc2 = nn.Linear(100, 10)
        self.out = nn.Linear(80, 2)

    def forward(self, x):
        x = self.conv1(x.unsqueeze(1))
        #print(x.size())
        x = self.conv2(x)
        #print(x.size())
        x = x.view(x.size(0), -1)
        x = self.lc1(x)
        #x = self.lc2(x)
        x = self.out(x)
        return x


net = Net().cuda()
print(net)
bsize = 500
log_train, log_test, label_train, label_test = getdata.get()
Train_data = Data.TensorDataset(log_train, label_train)
Test_data = Data.TensorDataset(log_test, label_test)
train_data = Data.DataLoader(dataset=Train_data, batch_size=bsize, shuffle=False)
test_data = Data.DataLoader(dataset=Test_data, batch_size=bsize, shuffle=False)
optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.5, weight_decay=1e-9)
loss_function = nn.CrossEntropyLoss()
for epoch in range(1000):

    #print("Epoch: {}".format(epoch))
    running_loss = 0.0
    for data in tqdm(train_data):
        inputs, labels = data
        optimizer.zero_grad()
        outputs = net(inputs.float())
        loss = loss_function(outputs, labels.long())
        loss.backward()
        optimizer.step()
        running_loss += loss.data.item()

    print('Epoch: {} | Loss: {}'.format(epoch, running_loss/label_train.size(0)))

print("Finished Training")

# TEST
correct = 0.0
total = 0
for data in test_data:
    log, labels = data
    outputs = net(log.float())
    _, predicted = t.max(outputs.data, 1)
    total += labels.size(0)
    for i in range(labels.size(0)):
        if predicted[i] == labels[i].long():
            correct = correct+1

print("Accuracy: {}".format(correct / total))

print("Dumping weights to disk")
weights_dict = {}
for param in list(net.named_parameters()):
    print("Serializing Param", param[0])
    weights_dict[param[0]] = param[1]
with open("../data/weights_new.pkl", "wb") as f:
    import pickle
    pickle.dump(weights_dict, f)
print("Finished dumping to disk..")
