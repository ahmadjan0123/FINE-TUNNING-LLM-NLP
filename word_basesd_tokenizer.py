import nltk
nltk.download("punkt_tab")


from nltk.tokenize import word_tokenize

text = 'this is is my life and it is all about studying.'
tokenizer = word_tokenize(text)
print(tokenizer)


text1 = 'I am Ahmad Yousuf Jan'
tokenizer1 = word_tokenize(text1)
print(tokenizer1)