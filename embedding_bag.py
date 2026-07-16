import torch
import torch.nn as nn

dataset = [
    'AI will change the world',
    'AI will change every industry',
    'AI will lead to AGI'
]
voc = {

}
for x in dataset:
    s = x.split()

    for z in s:
        if z in voc:
            voc[z] += 1
        else:
            voc[z] = 1


print(voc)

doc1 = [voc[word] for word in dataset[0].split()]
doc2 = [voc[word] for word in dataset[1].split()]
doc3 = [voc[word] for word in dataset[2].split()]

all_tokens = torch.tensor(doc1+doc2+doc3)


offsets = torch.tensor([0,len(doc1),len(doc1)+len(doc2)])


embedding_bag = nn.EmbeddingBag(
    num_embeddings = len(voc),
    embedding_dim = 4,
    mode = 'mean'
)


document_embedding = embedding_bag(all_tokens,offsets)

print('ALL TOEKNS',all_tokens)

print('offsets: ',offsets)

print('document Embedding',document_embedding)