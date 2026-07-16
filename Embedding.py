import torch 
import torch.nn as nn


dataset = [
    'i like apples',
    'i go for cycling'
]

vocab = {}

for sentences in dataset:
    x = sentences.split()

    for z in x:
        if z in vocab:
            vocab[z]+=1
        else:
            vocab[z] = 1


token = [vocab[word] for word in dataset[0].split()]

embedding = nn.Embedding(
    num_embeddings=5,
    embedding_dim=4
)


token = torch.tensor(token)

token = embedding(token)

print(token)