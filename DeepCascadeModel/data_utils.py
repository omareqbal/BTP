import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
import torch
import torch.utils as utils
from tqdm import tqdm
from allennlp.modules.elmo import batch_to_ids


def load_data(input_file, thres, max_plen):
	df = pd.read_json(input_file)
	data_ = df.to_dict('records')
	data = []

	for row in tqdm(data_):				
		if(row['max_score'] > thres and row['end'] < max_plen-2):
			data.append(row)

	return data



class Dataset(utils.data.Dataset):
	def __init__(self, data, max_plen, max_qlen, data_dir, glove_vec_size):
		super().__init__()
		self.data = data
		self.max_plen = max_plen
		self.max_qlen = max_qlen
		self.data_dir = data_dir
		self.glove_vec_size = glove_vec_size
		self.word2vec = self.load_glove_model()

	def __len__(self):
		return len(self.data)

	def __getitem__(self, index):
		row = self.data[index]
		q_words = word_tokenize(row['query'])
		e_q = self.get_glove_embedding(q_words, self.max_qlen)
		ch_q = self.get_elmo_charids(q_words, self.max_qlen)

		seq_len = []		

		e_d = []
		ch_d = []

		p_labels = []
		for p in row['passages']:
			if(p['is_selected'] == 1):
				p_labels.append(1)
			else:
				p_labels.append(0)

			p_words = word_tokenize(p['passage_text'])
			e_d_tmp = self.get_glove_embedding(p_words, self.max_plen)
			e_d.append(e_d_tmp)

			ch_d_tmp = self.get_elmo_charids(p_words, self.max_plen)
			ch_d.append(ch_d_tmp)

			seq_len.append(min(self.max_plen, len(p_words)+2))


		e_d = torch.stack(e_d)
		ch_d = torch.stack(ch_d)

		seq_len = torch.tensor(seq_len)
		p_labels = torch.Tensor(p_labels)
		qseq_len = min(self.max_qlen, len(q_words)+2)

		start = torch.tensor([row['start']+1])
		end = torch.tensor([row['end']+1])
		selected = torch.tensor([row['selected']])


		return row['query_id'], e_q, e_d, ch_q, ch_d, qseq_len, seq_len, p_labels, start, end, selected


	def load_glove_model(self):
	    print("Loading Glove Model")
	    glove_path = self.data_dir + '/glove/glove.6B.' + str(self.glove_vec_size) + 'd.txt'
	    f = open(glove_path,'r')
	    emb = {}
	    for line in f:
	        splitLine = line.split()
	        word = splitLine[0]
	        embedding = torch.tensor([float(val) for val in splitLine[1:]])
	        emb[word] = embedding
	    print("Done.",len(emb)," words loaded!")
	    f.close()
	    return emb


	def get_glove_embedding(self, words, max_len):
		e = torch.zeros(max_len, self.glove_vec_size)	
		for i, word in enumerate(words):
			if(i >= max_len-2):
				break
			word = word.lower()
			if(word in self.word2vec):
				e[i+1] = self.word2vec[word]

		temp = torch.zeros(max_len, 2)
		e = torch.cat([temp, e], dim=1)
		e[0][0] = 1								# <START>
		e[min(max_len,len(words)+2)-1][1] = 1	# <END>

		return e


	def get_elmo_charids(self, words_tmp, max_len):
		words = []
		words.append('<S>')
		words.extend(words_tmp)
		words.append('</S>')

		char_ids = batch_to_ids([words])[0]
		c = torch.zeros(max_len, char_ids.shape[1])
		for i in range(len(words)):
			if(i >= max_len):
				break
			c[i] = char_ids[i]

		return c