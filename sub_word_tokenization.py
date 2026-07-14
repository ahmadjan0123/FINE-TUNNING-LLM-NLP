from transformers import AutoTokenizer


tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

t1 = 'i am becaming the best ai engineer'
tokens = tokenizer.tokenize(t1)
print(tokens)


t2 = 'i am unstopable'
tokens = tokenizer.tokenize(t2)
ids1 = tokenizer.convert_tokens_to_ids(tokens)
print(tokens)
print(ids1)


t3 = 'i will do it'
token3 = tokenizer.tokenize(t3)
ids3 = tokenizer.convert_tokens_to_ids(token3)
ids3 = tokenizer.encode(t3)
print(ids3)


t4 = 'i will make it to top'
tokens4 = tokenizer.tokenize(t4)
ids4 = tokenizer.convert_tokens_to_ids(tokens4)

ids4 = tokenizer.encode(t4)
decoder = tokenizer.decode(ids4)

print(ids4)
print(decoder)