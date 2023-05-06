import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from preprocess_utils import *


def text_to_tensor(path, token2id, max_len=50):
    """returns two arrays
       
    contents is the array of token ids of the text contents,
    each row consists of a sentence, and each columns stores
    the corresponding token id of the tokens in the sentence.
    labels is the array of label of each sentence.
    """
    # texts = np.zeros(max_len)
    texts = []
    labels = []
    with open(path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            tokens = line.strip().split()
            labels.append(int(tokens[0]))
            sentence = []
            for token in tokens[1:]:
                if len(sentence) < max_len:
                    sentence.append(token2id.get(token, 0))
            pad_length = max(0, max_len - len(sentence))
            sentence += [0] * pad_length
            texts.append(sentence) # append the sentence to the contents
    texts = np.asarray(texts)
    texts = torch.from_numpy(texts).type(torch.int64)
    labels = np.asarray(labels)
    labels = torch.from_numpy(labels).type(torch.int64)
    return texts, labels


# get dataloader
def get_dataloader(batch_size, max_len):
    """returns three dataloaders for train, validation and test datasets"""
    token2id = build_token2id_dict()

    train_texts, train_labels = text_to_tensor('Dataset/train.txt', token2id, max_len)
    validation_texts, validation_labels = text_to_tensor('Dataset/validation.txt', token2id, max_len)
    test_texts, test_labels = text_to_tensor('Dataset/test.txt', token2id, max_len)

    train_dataset = TensorDataset(train_texts, train_labels)
    validation_dataset = TensorDataset(validation_texts, validation_labels)
    test_dataset = TensorDataset(test_texts, test_labels)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    validation_dataloader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    return train_dataloader, validation_dataloader, test_dataloader
