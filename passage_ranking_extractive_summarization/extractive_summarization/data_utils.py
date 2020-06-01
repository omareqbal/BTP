import pandas as pd
from nltk.tokenize import word_tokenize, sent_tokenize
import torch
import torch.utils as utils


def my_lcs(string, sub):
	# Calculates longest common subsequence for a pair of tokenized strings

    if(len(string)< len(sub)):
        sub, string = string, sub

    lengths = [[0 for i in range(0,len(sub)+1)] for j in range(0,len(string)+1)]

    for j in range(1,len(sub)+1):
        for i in range(1,len(string)+1):
            if(string[i-1] == sub[j-1]):
                lengths[i][j] = lengths[i-1][j-1] + 1
            else:
                lengths[i][j] = max(lengths[i-1][j] , lengths[i][j-1])

    return lengths[len(string)][len(sub)]


def calc_rouge_score(candidate, refs):

	# Compute ROUGE-L score given one candidate and references for an image

       
    beta = 1.2
    prec = []
    rec = []

    # split into tokens
    token_c = word_tokenize(candidate[0])
    if(len(token_c) == 0):
    	return 0.0
    if(token_c[-1] == '.'):
    	token_c = token_c[0:-1]
    if(len(token_c) == 0):
    	return 0.0
    for reference in refs:
        # split into tokens
        token_r = word_tokenize(reference)
        if(len(token_r) == 0):
        	continue
        if(token_r[-1] == '.'):
        	token_r = token_r[0:-1]
        if(len(token_r) == 0):
        	continue
        # compute the longest common subsequence
        lcs = my_lcs(token_r, token_c)
        prec.append(lcs/float(len(token_c)))
        rec.append(lcs/float(len(token_r)))

    if(len(prec) != 0):
    	prec_max = max(prec)
    else:
    	prec_max = 0
    if(len(rec) != 0):
    	rec_max = max(rec)
    else:
    	rec_max = 0

    if(prec_max!=0 and rec_max !=0):
        score = ((1 + beta**2)*prec_max*rec_max)/float(rec_max + beta**2*prec_max)
    else:
        score = 0.0
    return score


def load_data(input_file):
	df = pd.read_json(input_file, lines=True)
	data_ = df.to_dict('records')
	data = []
	for row in data_:
		if(len(row['passages']) == 10):
			data.append(row)

	return data


class Dataset(utils.data.Dataset):
	def __init__(self, data, max_s, max_slen, max_qlen, topk, glove_vec_size, data_dir):
		super().__init__()
		self.data = data
		self.max_s = max_s
		self.max_slen = max_slen
		self.max_qlen = max_qlen
		self.topk = topk
		self.glove_vec_size = glove_vec_size
		self.word2vec = self.load_glove_model(data_dir + '/glove/glove.6B.' + str(glove_vec_size) + 'd.txt')

	def __len__(self):
		return len(self.data)

	def __getitem__(self, index):
		row = self.data[index]
		e_q, qseq_len = get_embedding(row['query'])

		e_s = []
		
		sents = []

		para = sorted(row['passages'], key=lambda x:x['score'], reverse=True)
		answer_present = 0
		for i in range(self.topk):
			if(para[i]['is_selected'] == 1):
				answer_present = 1

			sent = sent_tokenize(para[i]['passage_text'])
			sents.extend(sent)

		best_sent = 0
		max_score = 0
		for i, sent in enumerate(sents):
			score = calc_rouge_score([sent], row['answers'])
			if(score > max_score):
				max_score = score
				best_sent = i

		labels = torch.zeros(self.max_s)
		sent_mask = torch.zeros(self.max_s)
		for i in range(len(sents)):
			if(i >= self.max_s):
				break
			if(i == best_sent and answer_present == 1):
				labels[i] = 1
			sent_mask[i] = 1

		e_s, seq_len = get_sent_embedding(sents)

		seq_len = torch.tensor(seq_len)
		n_sents = min(self.max_s, len(sents))

		return row['query_id'], e_q, e_s, qseq_len, n_sents, seq_len, labels, sent_mask


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


	def get_embedding(sent):
		words = word_tokenize(sent)
		e = torch.zeros(self.max_qlen, self.glove_vec_size)	
		for i, word in enumerate(words):
			if(i >= self.max_qlen):
				break
			word = word.lower()
			if(word in self.word2vec):
				e[i] = self.word2vec[word]

		qseq_len = min(self.max_qlen, len(words))
		return e, qseq_len


	def get_sent_embedding(sents):
		seq_len = [1 for i in range(max_s)]
		e = torch.zeros(self.max_s, self.max_slen, self.glove_vec_size)
		for i, sent in enumerate(sents):
			if(i >= max_s):
				break

			words = word_tokenize(sent)
			seq_len[i] = min(self.max_slen, len(words))

			for j, word in enumerate(words):
				if(j >= self.max_slen):
					break
				word = word.lower()
				if(word in self.word2vec):
					e[i][j] = self.word2vec[word]

		return e, seq_len
		