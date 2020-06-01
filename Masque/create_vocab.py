import pandas as pd
from nltk import word_tokenize
from tqdm import tqdm
from data_utils import load_glove_model, load_data
from collections import Counter
import pickle


data_dir = '../data'

glove_vec_size = 300

vocab_size = 5000


word2vec = load_glove_model(data_dir, glove_vec_size)

data = load_data(data_dir + '/msmarco/train_v2.1.json')
print('Done loading Training data')


word_freq = Counter() 

for row in tqdm(data):
    q_words = word_tokenize(row['query']) 
    for word_ in q_words:
        word = word_.lower() 
        if(word in word2vec): 
            word_freq[word] += 1 

    for p in row['passages']: 
        p_words = word_tokenize(p['passage_text'])        
        for word_ in p_words: 
            word = word_.lower() 
            if(word in word2vec): 
                word_freq[word] += 1

vocab = sorted(word_freq.most_common(vocab_size))
print('Done creating vocab.')

f = open('/vocab.txt', 'w')
for i in vocab:
    print(i[0], file=f)

f.close()

with open(code_dir + '/word_counts.pickle','wb') as f2:
	pickle.dump(word_freq, f2)