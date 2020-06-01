import pandas as pd
from nltk.tokenize import word_tokenize
import torch
import torch.utils as utils


def load_data(input_file):
	df = pd.read_json(input_file)
	data_ = df.to_dict('records')
	data = []
	for row in data_:
		if(len(row['passages']) == 10):
			data.append(row)

	return data


#
#	Dataset class to generate tensors for each training example
#

class Dataset(utils.data.Dataset):
	def __init__(self, data, max_plen, max_qlen, glove_vec_size, data_dir):
		super().__init__()
		self.data = data
		self.max_plen = max_plen
		self.max_qlen = max_qlen
		self.glove_vec_size = glove_vec_size
		self.word2vec = self.load_glove_model(data_dir + '/glove/glove.6B.' + str(glove_vec_size) + 'd.txt')


	def __len__(self):
		return len(self.data)

	def __getitem__(self, index):
		row = self.data[index]
		e_q = get_embedding(row['query'], self.max_qlen)

		seq_len = []		

		e_d = []

		labels = []
		for p in row['passages']:
			if(p['is_selected'] == 1):
				labels.append(1)
			else:
				labels.append(0)

			e_d_tmp = get_embedding(p['passage_text'], self.max_plen)
			e_d.append(e_d_tmp)

			words = word_tokenize(p['passage_text'])
			seq_len.append(min(self.max_plen,len(words)))


		e_d = torch.stack(e_d)

		seq_len = torch.tensor(seq_len)
		labels = torch.Tensor(labels)
		words = word_tokenize(row['query'])
		qseq_len = min(self.max_qlen, len(words))

		return row['query_id'], e_q, e_d, qseq_len, seq_len, labels


	def load_glove_model(gloveFile):
	    print("Loading Glove Model")
	    f = open(gloveFile,'r')
	    emb = {}
	    for line in f:
	        splitLine = line.split()
	        word = splitLine[0]
	        embedding = torch.tensor([float(val) for val in splitLine[1:]])
	        emb[word] = embedding
	    print("Done.",len(emb)," words loaded!")
	    f.close()
	    return emb


	def get_embedding(sent, max_len):
		words = word_tokenize(sent)
		e = torch.zeros(max_len, self.glove_vec_size)	
		for i, word in enumerate(words):
			if(i >= max_len):
				break
			word = word.lower()
			if(word in self.word2vec):
				e[i] = self.sword2vec[word]

		return e
