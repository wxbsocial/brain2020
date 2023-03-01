import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import WeightedRandomSampler
from sklearn.model_selection import LeaveOneOut
import random
import numpy as np
import os
import threading

import utils
from d2l import torch as d2l

import matplotlib.pyplot as plt
#scikit-learn related imports
import sklearn
from sklearn.datasets import fetch_california_housing
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# pytorch relates imports
import torch
import torch.nn as nn
import torch.optim as optim

# imports from captum library
from captum.attr import LayerConductance, LayerActivation, LayerIntegratedGradients
from captum.attr import IntegratedGradients, DeepLift, GradientShap, NoiseTunnel, FeatureAblation



filename = "./data/rest/Data_remove_0.csv"
error_sub_file = 'output/rest/error_sub_remove_0.txt'
model_save_dir = "./model/rest/"
model_save_path = model_save_dir + "model-fold-{}.pth"

load_data_size = -1
batch_size = 8
num_epochs = 1000
learning_rate = 0.001
drop_rate = 0.5
imbalanced_ratio = 1.0
fil_num = 64
anim = False
parallel = False


target = False
#1,3,17,21
# 8,
# filter 6,19，20
target_folds = [1,3,6,8, 16,17,19,20,21,22,24,25,29,31,32,33,37,38,44,45,47,48,55,58,61,62,63,65,68,69,73,77,78,80,81,83,86,91]

torch.manual_seed(42)


class CopdModel(nn.Module):
    def __init__(self, in_size, drop_rate, fil_num):
        super(CopdModel, self).__init__()
        self.net = nn.Sequential(
                # nn.Dropout(drop_rate),
                # nn.Linear(in_size, fil_num),
                # nn.LeakyReLU(),
                # nn.Dropout(drop_rate),
                # nn.Linear(fil_num, 2),
                # nn.Softmax(dim=1)


                # nn.Dropout(drop_rate),
                # nn.Linear(in_size, fil_num),
                # nn.LeakyReLU(),
                # nn.Dropout(drop_rate),
                # nn.Linear(fil_num , fil_num // 4),
                # nn.LeakyReLU(),
                # nn.Dropout(drop_rate),
                # nn.Linear(fil_num // 4, 2),
                # nn.Softmax(dim=1)


                nn.Dropout(drop_rate),
                nn.Linear(in_size, fil_num),
                nn.LeakyReLU(),
                nn.Dropout(drop_rate),
                nn.Linear(fil_num, fil_num // 2),
                nn.LeakyReLU(),
                nn.Dropout(drop_rate),
                nn.Linear(fil_num // 2, fil_num // 4),
                nn.LeakyReLU(),
                nn.Dropout(drop_rate),
                nn.Linear(fil_num // 4, fil_num // 8),
                nn.LeakyReLU(),
                nn.Dropout(drop_rate),
                nn.Linear(fil_num // 8, 2),
                #nn.Softmax(dim=1)
        )

    def forward(self, X):
        return self.net(X)



# FEATURE_NAMES = ["alff4","alff7","alff10","reho14","reho15","vmhc9","vmhc10"]
FEATURE_NAMES = ["alff4","alff5","alff7","alff10","reho6","reho14","reho15"]

def read_csv_copd(filename, size=-1):
    data = pd.read_csv(filename,usecols=['未来一年后加重次数',"姓名", 'alff1',"alff2", "alff3","alff4","alff5","alff6","alff7","alff8","alff9","alff10",
    "reho1","reho2","reho3","reho4","reho5"	,"reho6","reho7","reho8","reho9","reho10","reho11","reho12","reho13","reho14","reho15",	
    "vmhc1","vmhc2","vmhc3","vmhc4","vmhc5","vmhc6","vmhc7","vmhc8","vmhc9","vmhc10"])
    
    if (size > -1):
        data = data[:size]
    
    names = ["{}-{}".format(a,b) for a,b in zip(data['姓名'] , data['未来一年后加重次数'])] 
    #print(names)

    labels = (data['未来一年后加重次数'] > 0).astype(np.int_)

    demors = []

    for fn in FEATURE_NAMES:
        f = data[fn]
        demors.append((f - f.mean()) / f.std())

    print("demors len:", len(demors))


    demors = pd.concat(demors, axis=1, ignore_index=True)

    return names, labels.values.tolist(), demors.values.tolist()


def del_error_sub():
    if(os.path.isfile(error_sub_file)): 
        os.remove(error_sub_file)

def save_error_sub(fold,name,label,acc):
    f = open(error_sub_file, 'a+')
    f.write('{}\t{}\t{}\t{:.6f}\n'.format(fold, name, label, acc))
    f.close()



def train():

    if not os.path.exists(model_save_dir):
        os.mkdir(model_save_dir)
        
    results = {}
    n,y,X = read_csv_copd(filename, load_data_size)

    loo = LeaveOneOut()

    threads = []

    for fold,(train_idx, test_idx) in enumerate(loo.split(X)):
        if (target and fold not in target_folds):
            continue
        
        if parallel:
            t = threading.Thread(target=train_model, args=(fold,X,y,n,train_idx, test_idx,results))
            threads.append(t)
            t.start()
        else:
            train_model(fold,X,y,n,train_idx, test_idx,results)
 
    if parallel:
        for t in threads:
            t.join()


def train_model(fold,X,y,n,train_idx, test_idx,results):
    #init data 
    X_train = [X[i] for i in train_idx]
    y_train = [y[i] for i in train_idx]
    n_train = [n[i] for i in train_idx]

    X_test = [X[i] for i in test_idx]
    y_test = [y[i] for i in test_idx]
    n_test = [n[i] for i in test_idx]

    print("n_test", n_test)

    train_datasets = CopdDataset(X_train, y_train,n_train)
    sample_weight, imbalanced_ratio = train_datasets.get_sample_weights()
    sampler = WeightedRandomSampler(sample_weight, len(sample_weight))
    train_iter = DataLoader(train_datasets, batch_size=batch_size, sampler=sampler)

    test_datasets = CopdDataset(X_test, y_test,n_test)
    test_iter = DataLoader(test_datasets, batch_size=1,shuffle=False)

    #init model
    network = CopdModel(len(FEATURE_NAMES),drop_rate,fil_num)
    optimizer = optim.Adam(network.parameters(), lr=learning_rate, betas=(0.5, 0.999))
    
    # criterion = nn.CrossEntropyLoss(weight=torch.Tensor([1, imbalanced_ratio])).cpu()
    criterion = nn.CrossEntropyLoss().cpu()

    if anim:
        animator = d2l.Animator(xlabel='epoch', xlim=[1, num_epochs],
                            legend=['tl', 'ta', 'vl', 'va'])

    valid_metric = d2l.Accumulator(3)
    #train
    for epoch in range(num_epochs):  # loop over the dataset multiple times
        train_metric = d2l.Accumulator(3)

        network.train()
        for inputs, labels, _ in train_iter:
            # forward pass
            outputs = network(inputs)
            # defining loss
            loss = criterion(outputs, labels)
            # zero the parameter gradients
            optimizer.zero_grad()
            # computing gradients
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                train_metric.add(loss * inputs.shape[0], d2l.accuracy(outputs, labels), inputs.shape[0])

    
        #valid
        
        with torch.no_grad():
            network.eval()
            for inputs, labels, _ in test_iter:

                inputs, labels = inputs, labels
                outputs = network(inputs)
                loss = criterion(outputs, labels)

                acc_outputs = nn.Softmax(dim=1)(outputs)
                acc = acc_outputs[0,labels[0]]
                valid_metric.add(loss * inputs.shape[0], acc * inputs.shape[0], inputs.shape[0])
        
        if anim:
            if (epoch % 50 == 0 or epoch + 1 == num_epochs):
                valid_l = valid_metric[0] / valid_metric[2]
                valid_acc = valid_metric[1] / valid_metric[2]
                train_l = train_metric[0] / train_metric[2]
                train_acc = train_metric[1] / train_metric[2]
                animator.add(epoch + 1, (train_l, train_acc,valid_l, valid_acc))
    
    valid_l = valid_metric[0] / valid_metric[2]
    valid_acc = valid_metric[1] / valid_metric[2]
    print("loss={:.3f}, acc={:.3f}".format(valid_l, valid_acc))
    # Process is complete.
    print('Training process has finished. Saving trained model.')
    
    # Saving the model
    save_path = model_save_path.format(fold)
    torch.save(network.state_dict(), save_path)


def result(data_size = load_data_size):
   
    del_error_sub()

   
    actuals = []
    probabilities = []
    results = {}

    n,y,X = read_csv_copd(filename, data_size)
    loo = LeaveOneOut()
    for fold,(train_idx, test_idx) in enumerate(loo.split(X)):
        if (target and fold not in target_folds):
            continue

        #init data 
        X_test = [X[i] for i in test_idx]
        y_test = [y[i] for i in test_idx]
        n_test = [n[i] for i in test_idx]

        test_datasets = CopdDataset(X_test, y_test,n_test)
        test_iter = DataLoader(test_datasets, batch_size=1,shuffle=False)

        #load model
        model_path = model_save_path.format(fold)
        network = CopdModel(len(FEATURE_NAMES),drop_rate,fil_num)
        network.load_state_dict(torch.load(model_path))
        
        #test
        test_metric = d2l.Accumulator(2)
        with torch.no_grad():
            network.train(False)
            
            for inputs, labels,names in test_iter:

                inputs, labels = inputs, labels
                outputs = network(inputs)

                acc_outputs = nn.Softmax(dim=1)(outputs)
                acc = acc_outputs[0,labels[0]]
                test_metric.add(acc , 1)

                actuals.append(labels.numpy()[0])
                probabilities.append(acc_outputs[0,1].item())

                if (acc <= 0.5):
                    name = names[0]
                    real_label = labels[0]
                    save_error_sub(fold, name,real_label, acc)                     
        
        test_acc = test_metric[0] / test_metric[1]
        results[fold] = test_acc * 100       

    # Print fold results
    print(f'K-FOLD CROSS VALIDATION RESULTS FOR 0 FOLDS')
    print('--------------------------------')
    sum = 0.0
    for key, value in results.items():
        print(f'Fold {key}: {value} %')
        sum += value
    print(f'Average: {sum/len(results.items())} %')
    utils.draw_roc([actuals],[probabilities], 1,"ROC of rest", "rest")



def explain(data_size = load_data_size):

    x_axis_data = np.arange(len(FEATURE_NAMES))
    x_axis_data_labels = list(map(lambda idx: FEATURE_NAMES[idx], x_axis_data))

    ig_attr_test_sum = np.zeros_like(FEATURE_NAMES, dtype=np.float_)
    ig_attr_test_norm_sum = np.zeros_like(FEATURE_NAMES, dtype=np.float_)

    ig_nt_attr_test_sum = np.zeros_like(FEATURE_NAMES, dtype=np.float_)
    ig_nt_attr_test_norm_sum = np.zeros_like(FEATURE_NAMES, dtype=np.float_)

    dl_attr_test_sum = np.zeros_like(FEATURE_NAMES, dtype=np.float_)
    dl_attr_test_norm_sum = np.zeros_like(FEATURE_NAMES, dtype=np.float_)

    gs_attr_test_sum = np.zeros_like(FEATURE_NAMES, dtype=np.float_)
    gs_attr_test_norm_sum = np.zeros_like(FEATURE_NAMES, dtype=np.float_)

    fa_attr_test_sum = np.zeros_like(FEATURE_NAMES, dtype=np.float_)
    fa_attr_test_norm_sum = np.zeros_like(FEATURE_NAMES, dtype=np.float_)

    lin_weight = np.zeros_like(FEATURE_NAMES, dtype=np.float_)
    y_axis_lin_weight = np.zeros_like(FEATURE_NAMES, dtype=np.float_)

    n,y,X = read_csv_copd(filename, data_size)
    loo = LeaveOneOut()
    for fold,(train_idx, test_idx) in enumerate(loo.split(X)):
        if (target and fold not in target_folds):
            continue
        #init data 
        X_test = [X[i] for i in test_idx]
        y_test = [y[i] for i in test_idx]
        n_test = [n[i] for i in test_idx]

        X_test = torch.tensor(X_test)
        y_test = torch.tensor(y_test)    

        test_datasets = CopdDataset(X_test, y_test,n_test)
        test_iter = DataLoader(test_datasets, batch_size=1,shuffle=False)

        #load model
        model_path = model_save_path.format(fold)
        network = CopdModel(len(FEATURE_NAMES),drop_rate,fil_num)
        network.load_state_dict(torch.load(model_path))
        
        #test
        ig = IntegratedGradients(network)
        ig_nt = NoiseTunnel(ig)
        dl = DeepLift(network)
        gs = GradientShap(network)
        fa = FeatureAblation(network)

        target_class_index = 1
        data = X_test
        baselines = torch.zeros(data.shape)
        ig_attr_test = ig.attribute(data, target=target_class_index,n_steps=8)
        ig_nt_attr_test = ig_nt.attribute(data, target=target_class_index)
        dl_attr_test = dl.attribute(data, target=target_class_index)
        gs_attr_test = gs.attribute(data, baselines, target=target_class_index)
        fa_attr_test = fa.attribute(data, target=target_class_index)  

        ig_attr_test_sum += ig_attr_test.clone().detach().numpy().sum(0)
        ig_attr_test_norm_sum += ig_attr_test_sum / np.linalg.norm(ig_attr_test_sum, ord=1)

        ig_nt_attr_test_sum += ig_nt_attr_test.clone().detach().numpy().sum(0)
        ig_nt_attr_test_norm_sum += ig_nt_attr_test_sum / np.linalg.norm(ig_nt_attr_test_sum, ord=1)

        dl_attr_test_sum += dl_attr_test.clone().detach().numpy().sum(0)
        dl_attr_test_norm_sum += dl_attr_test_sum / np.linalg.norm(dl_attr_test_sum, ord=1)

        gs_attr_test_sum += gs_attr_test.clone().detach().numpy().sum(0)
        gs_attr_test_norm_sum += gs_attr_test_sum / np.linalg.norm(gs_attr_test_sum, ord=1)

        fa_attr_test_sum += fa_attr_test.clone().detach().numpy().sum(0)
        fa_attr_test_norm_sum += fa_attr_test_sum / np.linalg.norm(fa_attr_test_sum, ord=1)

        lin_weight += network.net[1].weight[0].clone().detach().numpy()
        y_axis_lin_weight += lin_weight / np.linalg.norm(lin_weight, ord=1)

    ig_attr_test_sum /= data_size
    ig_attr_test_norm_sum /= data_size

    ig_nt_attr_test_sum /= data_size
    ig_nt_attr_test_norm_sum /= data_size

    dl_attr_test_sum /= data_size
    dl_attr_test_norm_sum /= data_size

    gs_attr_test_sum /= data_size
    gs_attr_test_norm_sum /= data_size

    fa_attr_test_sum /= data_size
    fa_attr_test_norm_sum /= data_size

    lin_weight /= data_size
    y_axis_lin_weight /= data_size

    
    width = 0.14
    legends = ['Int Grads', 'Int Grads w/SmoothGrad','DeepLift', 'GradientSHAP', 'Feature Ablation', 'Weights']

    plt.figure(figsize=(20, 10))

    ax = plt.subplot()
    ax.set_title('Comparing input feature importances across multiple algorithms and learned weights')
    ax.set_ylabel('Attributions')

    FONT_SIZE = 16
    plt.rc('font', size=FONT_SIZE)            # fontsize of the text sizes
    plt.rc('axes', titlesize=FONT_SIZE)       # fontsize of the axes title
    plt.rc('axes', labelsize=FONT_SIZE)       # fontsize of the x and y labels
    plt.rc('legend', fontsize=FONT_SIZE - 4)  # fontsize of the legend

    ax.bar(x_axis_data, ig_attr_test_norm_sum, width, align='center', alpha=0.8, color='#eb5e7c')
    ax.bar(x_axis_data + width, ig_nt_attr_test_norm_sum, width, align='center', alpha=0.7, color='#A90000')
    ax.bar(x_axis_data + 2 * width, dl_attr_test_norm_sum, width, align='center', alpha=0.6, color='#34b8e0')
    ax.bar(x_axis_data + 3 * width, gs_attr_test_norm_sum, width, align='center',  alpha=0.8, color='#4260f5')
    ax.bar(x_axis_data + 4 * width, fa_attr_test_norm_sum, width, align='center', alpha=1.0, color='#49ba81')
    ax.bar(x_axis_data + 5 * width, y_axis_lin_weight, width, align='center', alpha=1.0, color='grey')
    ax.autoscale_view()
    plt.tight_layout()

    ax.set_xticks(x_axis_data + 0.5)
    ax.set_xticklabels(x_axis_data_labels)

    plt.legend(legends, loc=3)
    plt.show()   

class CopdDataset(Dataset):
    def __init__(self, X, y,names):

        self.X = torch.Tensor(X)
        self.y = y
        self.names = names
        
    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx],self.names[idx]

    def get_sample_weights(self):
        count, count0, count1 = float(len(self.y)), float(self.y.count(0)), float(self.y.count(1))
        weights = [count / count0 if i == 0 else count / count1 for i in self.y]
        return weights, 1
        # w = len(self.y) /  (self.y.sum(dim=0) + 0.00001)
        # print("w", w)
        # return w, 1

 
if __name__ == "__main__":
    a = input("Enter a (t,r,e,tr): ")
    if (a == 't'):
        train()
    elif (a == 'r'):
        result()
    elif (a == 'e'):
        explain()
    elif (a == 'tr'):
        train()
        result()
    else:
        train()
        result()

