import sys
import numpy as np
import torch.nn as nn
import torch
from sklearn.metrics import roc_auc_score
from Intedata_L import Intedata_L
from modle import MSTCRB
from torch.utils.data import Dataset, DataLoader

# %%Hyperparameter
protein = "AGO1"
TRAIN_BATCH_SIZE = 200
TEST_BATCH_SIZE = 200
NUM_EPOCHS = 20
LR = 0.0005
LOG_INTERVAL = 20
modeling = MSTCRB
cuda_name = "cuda:0"
if len(sys.argv) > 3:
    cuda_name = "cuda:" + str(int(0))
print('cuda_name:', cuda_name)


# %%mothod
def cal_auc(true_arr, pred_arr):
    auc0 = roc_auc_score(true_arr, pred_arr)
    return auc0

def calculate_accuracy(true_arr, pred_arr):
    """
    计算分类准确率
    Args:
        true_labels (list or numpy array): 真实标签列表或数组
        predicted_labels (list or numpy array): 预测标签列表或数组

    Returns:
        float: 准确率（0.0 到 1.0 之间的值）
    """
    # 确保输入的真实标签和预测标签具有相同的长度
    assert len(true_arr) == len(pred_arr), "长度不一致"
    pred_arr = threshold_predictions(pred_arr)
    # 计算正确预测的样本数量
    correct_count = sum(1 for true, pred in zip(true_arr, pred_arr) if true == pred)
    # 计算总样本数量
    total_count = len(true_arr)
    # 计算准确率
    accuracy = correct_count / total_count if total_count > 0 else 0.0

    return accuracy

def precision_score(true_arr, pred_arr):
    assert len(true_arr) == len(pred_arr)
    pred_arr = threshold_predictions(pred_arr)
    TP = sum(1 for true, pred in zip(true_arr, pred_arr) if (true == 1) & (pred == 1))  # True Positive
    FP = sum(1 for true, pred in zip(true_arr, pred_arr) if (true == 0) & (pred == 1))  # False Positive
    if TP + FP == 0:
        precision = 0
    else:
        precision = TP / (TP + FP)
    return precision

def recall_score(true_arr, pred_arr):
    assert len(true_arr) == len(pred_arr)
    pred_arr = threshold_predictions(pred_arr)
    TP = sum(1 for true, pred in zip(true_arr, pred_arr) if (true == 1) & (pred == 1))  # True Positive
    FN = sum(1 for true, pred in zip(true_arr, pred_arr) if (true == 1) & (pred == 0))  # False Negative
    if TP + FN == 0:
        recall = 0
    else:
        recall = TP / (TP + FN)
    return recall

def f1_score(true_arr, pred_arr):
    assert len(true_arr) == len(pred_arr)
    precision = precision_score(true_arr, pred_arr)
    recall = recall_score(true_arr, pred_arr)
    if precision + recall == 0:
        f1 = 0
    else:
        f1 = 2 * (precision * recall) / (precision + recall)
    return f1

def threshold_predictions(pred_arr, threshold=0.5):
    """
    将预测概率转换为二元类别标签（0 或 1）
    Args:
        pred_arr (list or numpy array): 模型的预测概率值列表或数组
        threshold (float): 阈值，用于决定预测为正类（1）或负类（0），默认为 0.5

    Returns:
        list: 转换后的二元类别标签列表（0 或 1）
    """
    # 将预测概率值转换为二元类别标签
    pred_arr = [1 if pred >= threshold else 0 for pred in pred_arr]

    return pred_arr


def train(model, device, train_loader, optimizer, epoch):
    print('Training on {} samples...'.format(len(train_loader.dataset)))
    model.train()
    total_preds = torch.Tensor()
    total_labels = torch.Tensor()
    '''
    for batch_idx, data in enumerate(train_loader):
        print(batch_idx)
    '''
    for batch_index, data in enumerate(train_loader):
        data["Kmer"] = data["Kmer"].to(device)
        data["NCP"] = data["NCP"].to(device)
        data["DPCP"] = data["DPCP"].to(device)
        data["knf"] = data["knf"].to(device)
        data["pair"] = data["pair"].to(device)
        optimizer.zero_grad()
        output = model(data, epoch)
        total_labels = torch.cat((total_labels, data["Y"].view(-1, 1).cpu()), 0)
        total_preds = torch.cat((total_preds, output.cpu()), 0)
        loss = loss_fn(output, data["Y"].float().to(device))
        loss.backward()
        optimizer.step()
    return total_labels.detach().numpy().flatten(), total_preds.detach().numpy().flatten(), loss

def predicting(model, device, loader, epoch):
    model.eval()
    total_preds = torch.Tensor()
    total_labels = torch.Tensor()
    print('Make prediction for {} samples...'.format(len(loader.dataset)))
    with torch.no_grad():
        for data in loader:
            data["Kmer"] = data["Kmer"].to(device)
            data["NCP"] = data["NCP"].to(device)
            data["DPCP"] = data["DPCP"].to(device)
            data["knf"] = data["knf"].to(device)
            data["pair"] = data["pair"].to(device)
            output = model(data, epoch)
            total_labels = torch.cat((total_labels, data["Y"].view(-1, 1).cpu()), 0)
            total_preds = torch.cat((total_preds, output.cpu()), 0)
    return total_labels.numpy().flatten(), total_preds.numpy().flatten()


def mse(y, f):
    mse = ((y - f) ** 2).mean(axis=0)
    return mse


# %%data preparation
if __name__ == '__main__':
    n_train = len(Intedata_L(protein))
    split = n_train // 5
    for i in range(1):
        indices = np.random.choice(range(n_train), size=n_train, replace=False)
        train_sampler = torch.utils.data.sampler.SubsetRandomSampler(indices[split:])
        test_sampler = torch.utils.data.sampler.SubsetRandomSampler(indices[:split])
        train_loader = DataLoader(Intedata_L(protein), sampler=train_sampler, batch_size=TRAIN_BATCH_SIZE)
        test_loader = DataLoader(Intedata_L(protein), sampler=test_sampler, batch_size=TEST_BATCH_SIZE)
        device = torch.device(cuda_name if torch.cuda.is_available() else "cpu")
        model = modeling().to(device)
        loss_fn = nn.BCELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=LR)
        with open('./Datasets/linearRNA-RBP/' + protein + '/cTrain1.txt', 'a') as f0, open('./Datasets/linearRNA-RBP/' + protein + '/cTest1.txt', 'a') as f1:
            for epoch in range(NUM_EPOCHS):
                print('Epoch: ', epoch+1)
                GT, GP, loss = train(model, device, train_loader, optimizer, epoch + 1)
                G, P = predicting(model, device, test_loader, epoch + 1)

                auc0 = cal_auc(GT, GP)
                acc0 = calculate_accuracy(GT, GP)
                Pre0 = precision_score(GT, GP)
                Recall0 = recall_score(GT, GP)
                F10 = f1_score(GT, GP)
                auc1 = cal_auc(G, P)
                acc1 = calculate_accuracy(G, P)
                Pre1 = precision_score(G, P)
                Recall1 = recall_score(G, P)
                F11 = f1_score(G, P)
                f0.write(f'epoch: {epoch + 1}\t')
                f0.write(f'auc0: {auc0:.5f}\t')  # 将 auc0 格式化为小数点后四位
                f0.write(f'acc: {acc0:.5f}\t')  # 将 acc 格式化为小数点后四位
                f0.write(f'precision: {Pre0:.5f}\t')  # 将 precision 格式化为小数点后四位
                f0.write(f'recall: {Recall0:.5f}\t')  # 将 recall 格式化为小数点后四位
                f0.write(f'f1: {F10:.5f}\t')  # 将 f1 格式化为小数点后四位
                f0.write(f'loss: {loss:.5f}\n')
                f1.write(f'epoch: {epoch + 1}\t')
                f1.write(f'auc0: {auc1:.5f}\t')  # 将 auc0 格式化为小数点后四位
                f1.write(f'acc: {acc1:.5f}\t')  # 将 acc 格式化为小数点后四位
                f1.write(f'precision: {Pre1:.5f}\t')  # 将 precision 格式化为小数点后四位
                f1.write(f'recall: {Recall1:.5f}\t')  # 将 recall 格式化为小数点后四位
                f1.write(f'f1: {F11:.5f}\n')  # 将 f1 格式化为小数点后四位
            f0.write(f'\n')
            f1.write(f'\n')

