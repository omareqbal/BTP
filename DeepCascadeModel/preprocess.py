import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
import torch
import torch.utils as utils
from tqdm import tqdm
import os
import argparse


def calc_lcs(string, sub):
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
        lcs = calc_lcs(token_r, token_c)
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



class Dataset(utils.data.Dataset):
	def __init__(self, data):
		super().__init__()
		self.data = data


	def __len__(self):
		return len(self.data)

	def __getitem__(self, index):
		row = self.data[index]

		max_score = 0
		start = -1
		end = -1
		selected = -1

		for j, p in enumerate(row['passages']):
			p_words = word_tokenize(p['passage_text'])

			p_start1 = 0
			p_end1 = 0
			p_max_score1 = 0

			for en in range(len(p_words)):
				span = " ".join(p_words[p_start1:en+1])
				score = calc_rouge_score([span], row['answers'])
				if(score > p_max_score1):
					p_end1 = en
					p_max_score1 = score

			for st in range(p_end1+1):
				span = " ".join(p_words[st:p_end1+1])
				score = calc_rouge_score([span], row['answers'])
				if(score > p_max_score1):
					p_start1 = st
					p_max_score1 = score


			p_start2 = 0
			p_end2 = len(p_words)-1
			p_max_score2 = 0

			
			for st in range(len(p_words)):
				span = " ".join(p_words[st:p_end2+1])
				score = calc_rouge_score([span], row['answers'])
				if(score > p_max_score2):
					p_start2 = st
					p_max_score2 = score


			for en in range(p_start2, len(p_words)):
				span = " ".join(p_words[p_start2:en+1])
				score = calc_rouge_score([span], row['answers'])
				if(score > p_max_score2):
					p_end2 = en
					p_max_score2 = score


			if(p_max_score1 >= p_max_score2):
				p_start = p_start1
				p_end = p_end1
				p_max_score = p_max_score1
			else:
				p_start = p_start2
				p_end = p_end2
				p_max_score = p_max_score2


			if(p_max_score > max_score):
				start = p_start
				end = p_end
				max_score = p_max_score
				selected = j

		answers = ('#####').join(row['answers'])

		return row, answers, start, end, selected, max_score


def load_data(input_file):
	df = pd.read_json(input_file)
	data_ = df.to_dict('records')
	data = []
	
	for i in tqdm(range(len(data_))):
		row = data_[i]
		if(len(row['passages']) == 10 and row['answers'] != ['No Answer Present.']):
			data.append(row)

	return data


parser = argparse.ArgumentParser()
parser.add_argument('--data', help='dataset to preprocess - train | dev')

args = parser.parse_args()


data_dir = '../data'

data = load_data(data_dir + '/msmarco/' + args.data + '_v2.1.json')

print('Done Loading data.')

params = {'batch_size': 16,
          'shuffle': False,
          'num_workers': 32,
	  	  'pin_memory': True}


train_set = Dataset(data)
train_generator = utils.data.DataLoader(train_set, **params)


data2 = []
for row, answers, start, end, selected, max_score in tqdm(train_generator):
	for i in range(len(start)):

		tmp = {}
		tmp['query_id'] = int(row['query_id'][i])
		tmp['query'] = row['query'][i]
		tmp['answers'] = answers[i].split('#####')
		tmp['passages'] = []
		for n,p in enumerate(row['passages']):
			tmp2 = {}
			tmp2['passage_text'] = p['passage_text'][i]
			tmp2['is_selected'] = int(p['is_selected'][i])
			tmp['passages'].append(tmp2)

		tmp['start'] = int(start[i])
		tmp['end'] = int(end[i])
		tmp['selected'] = int(selected[i])
		tmp['max_score'] = float(max_score[i])

		data2.append(tmp)


print('len data = ',len(data2))
df2 = pd.DataFrame(data2)

if not os.path.exists('preprocessed_data'):
	os.makedirs('preprocessed_data')

df2.to_json('preprocessed_data/' + args.data +'_data.json', orient='columns', default_handler=str)