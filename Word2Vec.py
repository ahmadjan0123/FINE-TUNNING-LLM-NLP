#py -3.12 "e:\11.Gen-AI for nlp and language understanding\Word2Vec.py"
from gensim.models import Word2Vec

sentences = [
    ['i',' love',' artificial','intelligence'],
    ['i' ,'love', 'machine', 'learning'],
    ['i' ,'love' ,'to' ,'eat' ,'pizza', 'burger', 'fries'],
    ['i', 'love', 'to', 'go', 'for', 'picninc']
]

print(sentences)

model = Word2Vec(
    sentences,
    vector_size=500,
    window=5,
    min_count=1,
    workers=4,
    epochs=500
)

model.save('WORDVEC.py')


# to find vectors 
print(model.wv['intelligence'])

#most similar
print('MOST SIMILAR TO LEARNNING \n')
print(model.wv.most_similar("learning"))

#check similarity between 2 words
print(model.wv.most_similar('learning','machine'))