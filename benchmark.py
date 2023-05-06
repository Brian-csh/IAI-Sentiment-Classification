import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from torch.optim.lr_scheduler import *
from sklearn.metrics import f1_score
import numpy as np
import argparse
from models import *
from preprocess_utils import *
from load_utils import *


def eval(dataloader):
    model.eval() # evaluation mode
    eval_loss = 0.0
    eval_acc = 0.0
    count = 0
    correct = 0
    true_labels = []
    pred_labels = []
    for i , (input, label) in enumerate(dataloader):
        output = model(input)
        loss = criterion(output, label)
        eval_loss += loss.item()
        correct += (output.argmax(1) == label).float().sum().item()
        count += len(input)
        true_labels.extend(label.cpu().numpy().tolist())
        pred_labels.extend(output.argmax(1).cpu().numpy().tolist())
    eval_loss = eval_loss * batch_size / len(dataloader.dataset)
    eval_acc = correct / count
    f1 = f1_score(np.array(true_labels), np.array(pred_labels), average='binary')
    return eval_loss, eval_acc, f1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Sentiment Classification Eval')
    parser.add_argument("model_type", help="CNN or LSTM or MLP or GRU")
    parser.add_argument("model_path", help="path to model")
    args = parser.parse_args()

    model_name = args.model_type
    model_path = args.model_path
    if model_name == "CNN":
        config = CNNConfig()
        model = CNN(config)
    elif model_name == "LSTM":
        config = LSTMConfig()
        model = LSTM(config)
    elif model_name == "MLP":
        config = MLPConfig()
        model = MLP(config)
    elif model_name == "GRU":
        config = GRUConfig()
        model = GRU(config)
    else:
        exit(1)
        
    model.load_state_dict(torch.load(model_path))
    
    batch_size = 50
    max_len = 120
    criterion = nn.CrossEntropyLoss()
    _, _, test_dataloader = get_dataloader(batch_size, max_len)
    test_loss, test_acc, test_f1 = eval(test_dataloader)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of trainable parameters: {num_params}")
    print(f"test_loss: {test_loss:.4f}, test_acc: {test_acc:.4f}, test_f1: {test_f1:.4f}")