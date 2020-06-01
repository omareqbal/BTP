import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
import torch
import torch.utils as utils
from tqdm import tqdm
from allennlp.modules.elmo import batch_to_ids
import numpy as np


def load_data(input_file, n_ans, n_no_ans):
	df = pd.read_json(input_file)
	data_ = df.to_dict('records')
	data_ans = []
	data_no_ans = []
	for i in tqdm(range(len(data_))):
		row = data_[i]
		if(len(row['passages']) == 10):
			if(row['answers'] != ['No Answer Present.']):
				data_ans.append(row)
			else:
				data_no_ans.append(row)

	data = []
	data.extend(data_ans[0:n_ans])
	data.extend(data_no_ans[0:n_no_ans])
	
	return data


def load_glove_model(data_dir, glove_vec_size):
    print("Loading Glove Model")
    glove_path = data_dir + '/glove/glove.6B.' + str(glove_vec_size) + 'd.txt'
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


def load_emb_matrix(vocab_file, glove_vec_size, glove_word2vec):
	f = open(vocab_file, 'r')
	vocab = f.readlines()
	f.close()
	emb_matrix = torch.zeros(len(vocab)+3, 2+glove_vec_size) # +3 for PAD, START, END tokens	
	word2idx = {}
	idx2word = {}

	emb_matrix[1][0] = 1  # <START>
	emb_matrix[2][1] = 1  # <END>

	word2idx['<PAD>'] = 0
	word2idx['<S>'] = 1
	word2idx['</S>'] = 2
	idx2word[0] = '<PAD>'
	idx2word[1] = '<S>'
	idx2word[2] = '</S>'

	for i, word_ in enumerate(vocab):
		word = word_[:-1]					# removing \n
		emb_matrix[i+3] = torch.cat([torch.zeros(2),glove_word2vec[word]])
		word2idx[word] = i+3
		idx2word[i+3] = word

	return emb_matrix, word2idx, idx2word


class Dataset(utils.data.Dataset):
	def __init__(self, data, max_plen, max_qlen, max_ans_len, glove_vec_size, glove_word2vec, common_word2idx, qp_vocab_size, mode, style=None):
		super().__init__()
		self.data = data
		self.max_plen = max_plen
		self.max_qlen = max_qlen
		self.max_ans_len = max_ans_len
		self.glove_vec_size = glove_vec_size
		self.glove_word2vec = glove_word2vec
		self.common_word2idx = common_word2idx
		self.common_vocab_words = set(common_word2idx.keys())
		self.common_vocab_size = len(self.common_vocab_words)
		self.qp_vocab_size = qp_vocab_size
		self.mode = mode
		self.style = style

	def __len__(self):
		return len(self.data)

	def __getitem__(self, index):
		row = self.data[index]
		q_words = word_tokenize(row['query'])
		q_words_lower = [w.lower() for w in q_words]

		e_q = self.get_glove_embedding(q_words_lower, self.max_qlen)

		ch_q = self.get_elmo_charids(q_words, self.max_qlen)

		qp_vocab = set()
		qp_vocab = self.add_to_qp_vocab(qp_vocab, q_words_lower, self.max_qlen)

		seq_len = []		

		e_p = []
		ch_p = []
		
		p_words_lower = []
		for p in row['passages']:

			p_words_tmp = word_tokenize(p['passage_text'])
			p_words_tmp_lower = [w.lower() for w in p_words_tmp]
			p_words_lower.append(p_words_tmp_lower)

			qp_vocab = self.add_to_qp_vocab(qp_vocab, p_words_tmp_lower, self.max_plen)

			e_p_tmp = self.get_glove_embedding(p_words_tmp_lower, self.max_plen)
			e_p.append(e_p_tmp)

			ch_p_tmp = self.get_elmo_charids(p_words_tmp, self.max_plen)
			ch_p.append(ch_p_tmp)

			seq_len.append(min(self.max_plen, len(p_words_tmp_lower)+2)) # START and END tokens


		e_p = torch.stack(e_p)
		ch_p = torch.stack(ch_p)

		seq_len = torch.tensor(seq_len)
		qseq_len = torch.tensor(min(self.max_qlen, len(q_words_lower)+2))

		if(self.mode == 'predict'):
			qp_vocab = sorted(list(qp_vocab))
			qp_idx2word, q_words_idx, p_words_idx = self.create_copy_dist_vocab(q_words_lower, p_words_lower, qp_vocab)
			return row['query_id'], e_q, e_p, ch_q, ch_p, qseq_len, seq_len, q_words_idx, p_words_idx, qp_idx2word


		p_labels = []

		for p in row['passages']:
			if(p['is_selected'] == 1):
				p_labels.append(1)
			else:
				p_labels.append(0)

		p_labels = torch.Tensor(p_labels)


		if(row['answers'] == ['No Answer Present.']):
			ans_pos = [0]
		else:
			ans_pos = [1]

		ans_pos = torch.Tensor(ans_pos)

		if(self.mode == 'dev'):
			if(self.style == '<NLG>' and row['wellFormedAnswers'] != '[]'):
				ans = row['wellFormedAnswers'][0]
			else:
				ans = row['answers'][0]
			style = self.style
		else:
			if(row['wellFormedAnswers'] != '[]' and np.random.random() > 0.5):
				ans = row['wellFormedAnswers'][0]
				style = '<NLG>'
			else:
				ans = row['answers'][0]
				style = '<QA>'

		ans_words = word_tokenize(ans)
		ans_words_lower = [w.lower() for w in ans_words]
		e_ans = self.get_ans_glove_embedding(ans_words_lower, style)

		qp_vocab = self.add_to_qp_vocab(qp_vocab, ans_words_lower, self.max_ans_len)

		qp_vocab = sorted(list(qp_vocab))

		ch_ans = self.get_ans_elmo_charids(ans_words, style)

		aseq_len = min(self.max_ans_len, len(ans_words_lower)+2)	# style and <START> tokens

		ans_mask = torch.ones(aseq_len)
		if(aseq_len < self.max_ans_len):
			ans_mask = torch.cat([ans_mask, torch.zeros(self.max_ans_len-aseq_len)], dim=0)

		aseq_len = torch.tensor(aseq_len)
		
		q_words_idx, p_words_idx, ans_words_idx = self.create_copy_dist_vocab(q_words_lower, p_words_lower, qp_vocab, ans_words_lower)

		return row['query_id'], e_q, e_p, ch_q, ch_p, qseq_len, seq_len, q_words_idx, p_words_idx, p_labels, ans_pos, e_ans, ch_ans, aseq_len, ans_words_idx, ans_mask
		

	def add_to_qp_vocab(self, qp_vocab, words, max_len):
		qp_vocab_tmp = qp_vocab.copy()
		for i, word in enumerate(words):
			if(i >= max_len-2):
				break
			if(word not in self.common_vocab_words):
				qp_vocab_tmp.add(word)
		return qp_vocab_tmp


	def create_copy_dist_vocab(self, q_words, p_words, qp_vocab, ans_words=None):
		qp_word2idx = {}
		qp_idx2word = []
		for i, word in enumerate(qp_vocab):
			qp_word2idx[word] = i + self.common_vocab_size
			qp_idx2word.append(word)

		qp_idx2word.extend(['' for _ in range(self.qp_vocab_size - len(qp_vocab))])

		q_words_idx = self.get_words_idx(q_words, self.max_qlen, qp_word2idx)

		num_para = len(p_words)
		p_words_idx = []
		for i in range(num_para):
			p_words_idx_tmp = self.get_words_idx(p_words[i], self.max_plen, qp_word2idx)
			p_words_idx.append(p_words_idx_tmp)

		p_words_idx = torch.stack(p_words_idx)


		if(self.mode == 'predict'):
			return qp_idx2word, q_words_idx, p_words_idx

		ans_words_idx = self.get_words_idx(ans_words, self.max_ans_len, qp_word2idx)
		
		return q_words_idx, p_words_idx, ans_words_idx


	def get_words_idx(self, words, max_len, qp_word2idx):
		words_idx = []
		words_idx.append(1)				# <START>
		for i, w in enumerate(words):
			if(i >= max_len-2):
				break
			if(w in self.common_vocab_words):
				words_idx.append(self.common_word2idx[w])
			else:
				words_idx.append(qp_word2idx[w])
		words_idx.append(2)				# <END>

		if(len(words_idx) < max_len):
			words_idx.extend([0 for _ in range(max_len-len(words)-2)])	# <PAD>

		return torch.tensor(words_idx)


	def get_glove_embedding(self, words, max_len):
		e = torch.zeros(max_len, self.glove_vec_size)	
		for i, word in enumerate(words):
			if(i >= max_len-2):
				break
			if(word in self.glove_word2vec):
				e[i+1] = self.glove_word2vec[word]

		temp = torch.zeros(max_len, 2)
		temp2 = torch.zeros(max_len, 2)
		e = torch.cat([temp, e, temp2], dim=1)
		e[0][0] = 1								# <START>
		e[min(max_len,len(words)+2)-1][1] = 1	# <END>

		return e


	def get_ans_glove_embedding(self, words, style):
		e = torch.zeros(self.max_ans_len, self.glove_vec_size)	
		for i, word in enumerate(words):
			if(i >= self.max_ans_len-2):
				break
			if(word in self.glove_word2vec):
				e[i+2] = self.glove_word2vec[word]

		temp = torch.zeros(self.max_ans_len, 2)
		temp2 = torch.zeros(self.max_ans_len, 2)
		e = torch.cat([temp, e, temp2], dim=1)
		e[1][0] = 1							# <START>

		if(style == '<QA>'):
			e[0][-2] = 1
		elif(style == '<NLG>'):
			e[0][-1] = 1

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

	def get_ans_elmo_charids(self, words_tmp, style):

		words = []
		words.append(style)
		words.append('<S>')
		words.extend(words_tmp)

		char_ids = batch_to_ids([words])[0]
		c = torch.zeros(self.max_ans_len, char_ids.shape[1])
		for i in range(len(words)):
			if(i >= self.max_ans_len):
				break
			c[i] = char_ids[i]

		return c
