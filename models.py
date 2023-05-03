import torch
import torch.nn as nn
import torch.nn.functional as F
from preprocess_utils import *


token2id = build_token2id_dict()
pretrained_embedding = get_embedding(token2id)
vocab_size = len(token2id) + 1


class CNNConfig(object):
    """Configuration for CNN model"""
    vocab_size = vocab_size
    embedding_dim = 50
    num_filters = 20
    kernel_sizes = [3, 4, 5]
    dropout = 0.3
    pretrained_embedding = pretrained_embedding


class CNN(nn.Module):
    """CNN model based on Kim, Y. (2014)"""
    def __init__(self, config):
        super(CNN, self).__init__()

        # setup embedding
        self.embedding = nn.Embedding(config.vocab_size, config.embedding_dim)
        self.embedding.weight.requires_grad = False
        self.embedding.weight.data.copy_(torch.from_numpy(config.pretrained_embedding))

        # convolution layers
        # kernel is of size k x embedding dim, with num_filters output channels
        # we use multiple filters with varying window sizes
        self.conv1 = nn.Conv2d(1, config.num_filters, (config.kernel_sizes[0], config.embedding_dim))
        self.conv2 = nn.Conv2d(1, config.num_filters, (config.kernel_sizes[1], config.embedding_dim))
        self.conv3 = nn.Conv2d(1, config.num_filters, (config.kernel_sizes[2], config.embedding_dim))
        # dropout rate
        self.dropout = nn.Dropout(config.dropout)
        # fully-connected layer
        self.fc = nn.Linear(config.num_filters * len(config.kernel_sizes), 2)
    

    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x)).squeeze(3)
        return F.max_pool1d(x, x.size(2)).squeeze(2)
    

    def forward(self, x):
        x = self.embedding(x)
        x = x.unsqueeze(1) # unsqueeze along 1 to match the format for conv2d

        x1 = self.conv_and_pool(x, self.conv1)
        x2 = self.conv_and_pool(x, self.conv2)
        x3 = self.conv_and_pool(x, self.conv3)

        x = torch.cat((x1, x2, x3), 1)
        x = self.dropout(x)

        logits = self.fc(x)

        return F.log_softmax(logits, dim=1)


class RNN_LSTMConfig(object):
    """Configuration for RNN_LSTM model"""
    vocab_size = vocab_size
    embedding_dim = 50
    hidden_size = 50
    num_hidden_layers = 2
    dropout = 0.3
    pretrained_embedding = pretrained_embedding


class RNN_LSTM(nn.Module):
    def __init__(self, config):
        super(RNN_LSTM, self).__init__()

        # setup embedding
        self.embedding = nn.Embedding(config.vocab_size, config.embedding_dim)
        self.embedding.weight.requires_grad = False
        self.embedding.weight.data.copy_(torch.from_numpy(config.pretrained_embedding))

        self.lstm = nn.LSTM(input_size=config.embedding_dim, hidden_size=config.hidden_size, num_layers=config.num_hidden_layers, bidirectional=True)
        
        self.fc1 = nn.Linear(config.hidden_size * 2, 64)
        self.fc2 = nn.Linear(64, 2)
        # todo: add dropout later and compare


    def forward(self, inputs):
        embedded_inputs = self.embedding(inputs) # embed
        _, (hn, _) = self.lstm(embedded_inputs.permute(1, 0, 2)) # permute to (seq_len, batch, input_size)
        hn = hn.view(self.num_layers, 2, -1, self.hidden_size)
        hn = torch.cat((hn[-1, 0], hn[-1, 1]), dim=-1)
        return self.fc2(self.fc1(hn))
        # todo: fix the comments


class RNN_GRUConfig(object):
    """Configuration for RNN_GRU model"""
    vocab_size = vocab_size
    embedding_dim = 50
    hidden_size = 50
    num_hidden_layers = 2
    dropout = 0.3
    pretrained_embedding = pretrained_embedding


