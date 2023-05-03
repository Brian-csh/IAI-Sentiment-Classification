import torch.nn as nn
from gensim.models import KeyedVectors
import numpy as np
import os


def build_token2id_dict():
    """build the vocabulary to assign a tokenid to each token in the corpus."""
    token2id = {}
    # files = ['sample.txt']
    files = ['Dataset/train.txt', 'Dataset/validation.txt', 'Dataset/test.txt']
    for file in files:
        with open(file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for line in lines:
                tokens = line.strip().split()
                for token in tokens[1:]:
                    if token not in token2id.keys():
                        token2id[token] = len(token2id)
    return token2id


def get_embedding(token2id):
    """get the pretrained embedding from word2vec and format it according to token id."""
    word2vec = KeyedVectors.load_word2vec_format('Dataset/wiki_word2vec_50.bin', binary=True)
    # set the parameters of the embedding
    embedding_parameters = np.zeros((len(token2id) + 1, word2vec.vector_size))
    
    for key in token2id:
        try:
            embedding_parameters[token2id[key]] = word2vec[key]
        except KeyError:
            pass

    # print(embedding_parameters[40])
    # print(word2vec['演得'])
    return embedding_parameters

# getEmbedding(get_tokenid_dict())
