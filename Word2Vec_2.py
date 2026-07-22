#py -3.12 "e:\11.Gen-AI for nlp and language understanding\Word2Vec_2.py"

from gensim.models import Word2Vec

sentences = [
    ['i',' love',' artificial','intelligence'],
    ['i' ,'love', 'machine', 'learning'],
    ['i' ,'love' ,'to' ,'eat' ,'pizza', 'burger', 'fries'],
    ['i', 'love', 'to', 'go', 'for', 'picnic']
]

model = Word2Vec(
    sentences,
    window=5,
    min_count=1,
    vector_size=10,
    workers=2,
    epochs=100 # when the training data is small 
)

print('vector represenatation of picnic',model.wv['picnic'])

print('Similarity between 2 words: ',model.wv.most_similar('picnic','learning'))