import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from torch.optim.lr_scheduler import *
from sklearn.metrics import f1_score
import numpy as np
import argparse
from tqdm import tqdm
from models import *
from preprocess_utils import *
from load_utils import *
from torch.utils.tensorboard import SummaryWriter


def save_model(model, step):
    path = 'logs/{}_{}.pt'.format(model_name, step)
    torch.save(model.state_dict(), path)


def train(dataloader):
    model.train() # train mode
    train_loss = 0.0
    train_acc = 0.0
    count = 0
    correct = 0
    true_labels = []
    pred_labels = []
    for i, (input, label) in enumerate(dataloader):
        optimizer.zero_grad()
        output = model(input) # forward pass
        loss = criterion(output, label) # calculate loss
        loss.backward() # compute gradient
        optimizer.step() # update parameters
        train_loss += loss.item()
        correct += (output.argmax(1) == label).float().sum().item()
        count += len(input)
        true_labels.extend(label.cpu().numpy().tolist())
        pred_labels.extend(output.argmax(1).cpu().numpy().tolist())
    train_loss = train_loss * batch_size / len(dataloader.dataset)
    train_acc = correct / count
    scheduler.step()
    f1 = f1_score(np.array(true_labels), np.array(pred_labels), average='binary')
    return train_loss, train_acc, f1


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
    parser = argparse.ArgumentParser(description='Sentiment Classification')
    parser.add_argument("--model", help="CNN or RNN_LSTM", default="CNN")
    parser.add_argument("--lr", type=float, default=0.001, help="learning rate")
    parser.add_argument("--epochs", type=int, default=10, help="epochs to train")
    parser.add_argument("--batch_size", type=int, default=50, help="batch size")
    parser.add_argument("--max_len", type=int, default=120, help="maximum sentence length")
    args = parser.parse_args()

    model_name = args.model
    if model_name == "RNN_LSTM":
        config = RNN_LSTMConfig()
        model = RNN_LSTM(config)
    else:
        config = CNNConfig()
        model = CNN(config)

    learning_rate = args.lr
    epochs = args.epochs
    batch_size = args.batch_size
    max_len = args.max_len
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    scheduler = StepLR(optimizer, step_size=5)

    train_dataloader, validation_dataloader, test_dataloader = get_dataloader(batch_size, max_len)

    writer = SummaryWriter()

    for step in tqdm(range(1, epochs+1)):
        train_loss, train_acc, train_f1 = train(train_dataloader)
        validation_loss, validation_acc, validation_f1 = eval(validation_dataloader)
        test_loss, test_acc, test_f1 = eval(test_dataloader)
        writer.add_scalar("Train Loss", train_loss, step)
        writer.add_scalar("Train Acc", train_acc, step)
        writer.add_scalar("Train f1", train_f1, step)
        writer.add_scalar("Validation Loss", validation_loss, step)
        writer.add_scalar("Validation Acc", validation_acc, step)
        writer.add_scalar("Validation f1", validation_f1, step)
        writer.add_scalar("Test Loss", test_loss, step)
        writer.add_scalar("Test Acc", test_acc, step)
        writer.add_scalar("Test f1", test_f1, step)
        print(f"Epoch {step}/{epochs}")
        print(f"Train loss: {train_loss:.4f}, Train acc: {train_acc:.4f}")
        print(f"Validation loss: {validation_loss:.4f}, Validation acc: {validation_acc:.4f}")
        print(f"Test loss: {test_loss:.4f}, Test acc: {test_acc:.4f}, Test f1: {test_f1:.4f}")
        save_model(model, step)

    writer.close()