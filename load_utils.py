import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from preprocess_utils import *


def load_data_from_text(path, token2id, max_len=50):
    """ returns two arrays
        contents is the array of token ids of the text contents,
            each row consists of a sentence, and each columns stores
            the corresponding token id of the tokens in the sentence.
        labels is the array of label of each sentence.
    """
    texts = np.zeros(max_len)
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
            sentence = np.asarray(sentence)
            pad_length = max(0, max_len - len(sentence))
            sentence = np.pad(sentence, (0, pad_length), 'constant', constant_values=0) # apply padding
            texts = np.vstack((texts, sentence)) # append the sentence to the contents
    texts = np.delete(texts, 0, axis=0) # delete the placeholder
    return texts, np.asarray(labels)


# get dataloader
def get_dataloader(batch_size, max_len):
    """ returns three dataloaders for train, validation and test datasets"""
    token2id = build_token2id_dict()

    train_texts, train_labels = load_data_from_text('Dataset/train.txt', token2id, max_len)
    validation_texts, validation_labels = load_data_from_text('Dataset/validation.txt', token2id, max_len)
    test_texts, test_labels = load_data_from_text('Dataset/test.txt', token2id, max_len)

    train_dataset = TensorDataset(torch.from_numpy(train_texts).type(torch.int64), torch.from_numpy(train_labels).type(torch.long))
    validation_dataset = TensorDataset(torch.from_numpy(validation_texts).type(torch.int64), torch.from_numpy(validation_labels).type(torch.long))
    test_dataset = TensorDataset(torch.from_numpy(test_texts).type(torch.int64), torch.from_numpy(test_labels).type(torch.long))

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    validation_dataloader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    return train_dataloader, validation_dataloader, test_dataloader
