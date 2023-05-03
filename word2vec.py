from gensim.models import KeyedVectors
import torch

w2v = KeyedVectors.load_word2vec_format('Dataset/wiki_word2vec_50.bin', binary=True)

w1 = torch.tensor(w2v["国王"])
w2 = torch.tensor(w2v["皇帝"])
w3 = torch.tensor(w2v["乞丐"])
print(torch.dot(w1, w2))
print(torch.dot(w1, w3))