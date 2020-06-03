from tqdm import tqdm
import torch
import torch.nn as nn
from modelfile import Model, Config
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
from data_utils import Dataset, load_data, load_glove_model, load_emb_matrix
import os
import json
import numpy as np


data_dir = '../data'

elmo_emb_size = 256
elmo_options = data_dir + '/elmo/options.json'
elmo_weights = data_dir + '/elmo/weights.hdf5'


DELTA = 0.00000000001
weight_decay = 0.01
lr = 0.0002

common_vocab_size = 5003 	# 3 for <START>, <END>, <PAD>
qp_vocab_size = 300			# vocab size for copy distribution

glove_vec_size = 300
hidden_size = 256

max_plen = 128				# Maximum passage length
max_qlen = 32				# Maximum query length
max_ans_len = 32			# Maximum answer length
num_para = 10				# Number of passages

n_attention_heads = 4
dropout = 0.3
dim_feedforward = 128
n_shared_encoder_blocks = 2
n_modeling_encoder_blocks_q = 2
n_modeling_encoder_blocks_p = 3
n_decoder_blocks = 5


ans_pos_thres = 0.4		# threshold for answer possibility classifier
style = '<QA>'


### <PAD> token - 	   [0,0,...,0,0]	... -> 300d glove
### <START> token -    [1,0,...,0,0]
### <END> token - 	   [0,1,...,0,0]

### QA Style Answer -  [0,0,...,1,0]
### NLG Style Answer - [0,0,...,0,1]

def get_ans_embeddings(ans_words, batch_size):
	e_ans = []
	ch_ans = []
	aseq_len = []
	ans_mask = []
	for i in range(batch_size):
		ans_words_lower = [w.lower() for w in ans_words[i]]
		e_ans_tmp = eval_set.get_ans_glove_embedding(ans_words_lower, style)
		ch_ans_tmp = eval_set.get_ans_elmo_charids(ans_words[i], style)
		aseq_len_tmp = min(max_ans_len, len(ans_words[i])+2)

		e_ans.append(e_ans_tmp)
		ch_ans.append(ch_ans_tmp)
		aseq_len.append(aseq_len_tmp)
		ans_mask_tmp = torch.ones(aseq_len_tmp)
		if(aseq_len_tmp < max_ans_len):
			ans_mask_tmp = torch.cat([ans_mask_tmp, torch.zeros(max_ans_len-aseq_len_tmp)], dim=0)
		ans_mask.append(ans_mask_tmp)

	e_ans = torch.stack(e_ans)
	ch_ans = torch.stack(ch_ans)
	aseq_len = torch.tensor(aseq_len)
	ans_mask = torch.stack(ans_mask_tmp)

	return e_ans, ch_ans, aseq_len, ans_mask


def predict():

	results = []

	model.eval()

	with torch.no_grad():

		for query_id, e_q, e_p, ch_q, ch_p, qseq_len, seq_len, q_words_idx, p_words_idx, qp_idx2word in tqdm(eval_generator):
						
			e_q =  e_q.to(device)
			e_p = e_p.to(device)
			ch_q = ch_q.long().to(device)
			ch_p = ch_p.long().to(device)
			qseq_len = qseq_len.to(device)
			q_words_idx = q_words_idx.long().to(device)
			seq_len = seq_len.to(device)
			p_words_idx = p_words_idx.long().to(device)


			batch_size = query_id.shape[0]

			beta_p, prob_ans, M_q, q_mask, M_p_k, p_mask = model.module.reader_ranker_classifier(e_q, e_p, ch_q, ch_p, qseq_len, seq_len)

			ans_words = []
			for i in range(batch_size):
				ans_words_tmp = []
				ans_words.append(ans_words_tmp)

			e_ans, ch_ans, aseq_len, ans_mask = get_ans_embeddings(ans_words, batch_size)


			for t in range(1, max_ans_len):
				e_ans = e_ans.to(device)
				ch_ans = ch_ans.long().to(device)
				aseq_len = aseq_len.to(device)

				P_y = model.module.ans_decoder(e_ans, ch_ans, aseq_len, M_q, q_mask, M_p_k, p_mask, beta_p, q_words_idx, p_words_idx)
				# shape of P_y - [batch_size, ans_len, ext_vocab_size]

				_, ans_word_idx = torch.topk(P_y, k=10, dim=2)	# select top-10 words

				complete = True
				completed = [False for _ in range(batch_size)]
				for i in range(batch_size):
					word_idx = int(ans_word_idx[i][t][np.random.randint(0,10)])
					if(word_idx < common_vocab_size):
						word = idx2word[word_idx]
					else:
						word = qp_idx2word[word_idx - common_vocab_size][i]


					if(word == '</S>'):
						completed[i] = True

					if(completed[i] == False and word != '<S>' and word != '</S>' and prob_ans[i] >= ans_pos_thres):
						ans_words[i].append(word)
						complete = False

				if(complete or t == max_ans_len-1):
					break

				e_ans, ch_ans, aseq_len, ans_mask = get_ans_embeddings(ans_words, batch_size)

			for i in range(batch_size):
				if(prob_ans[i] < ans_pos_thres):
					ans_words[i].extend(['No','Answer','Present.'])

			results.extend(ans_words)

	return results



glove_word2vec = load_glove_model(data_dir, glove_vec_size)

emb_matrix, word2idx, idx2word = load_emb_matrix('/vocab.txt', glove_vec_size, glove_word2vec)
print('Done loading emb_matrix.')

eval_data = load_data(data_dir + '/msmarco/dev_v2.1.json')[0:25000]

print('Done loading Eval data.')


eval_params = {'batch_size': 32,
          'shuffle': False,
          'num_workers': 32,
	  	  'pin_memory': True}


eval_set = Dataset(eval_data, max_plen, max_qlen, max_ans_len, glove_vec_size, glove_word2vec, word2idx, qp_vocab_size, mode='predict')
eval_generator = DataLoader(eval_set, **eval_params)



config = Config(glove_vec_size,
				elmo_options,
				elmo_weights,
				elmo_emb_size,
				common_vocab_size,
				qp_vocab_size,
				hidden_size, 
				max_plen, 
				max_qlen, 
				max_ans_len,
				num_para, 
				n_attention_heads,
				dropout,
				dim_feedforward,
				n_shared_encoder_blocks,
				n_modeling_encoder_blocks_q,
				n_modeling_encoder_blocks_p,
				n_decoder_blocks,
				device
			)

model = Model(config, emb_matrix)


if(cuda):
	model = model.to(device)
	model = nn.DataParallel(model)

optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

checkpoint = torch.load('checkpoints/saved_model.pth')

model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

if not os.path.exists('outputs'):
	os.makedirs('outputs')

results = predict()

f1 = open('outputs/dev_ref.json','w')
f2 = open('outputs/dev_pred.json','w')

for i in range(len(eval_data)):
	tmp1 = {}
	tmp1['query_id'] = eval_data[i]['query_id']
	tmp1['answers'] = eval_data[i]['answers']
	json.dump(tmp1,f1)
	f1.write('\n')

	tmp2 = {}
	tmp2['query_id'] = eval_data[i]['query_id']
	tmp2['answers'] = [' '.join(results[i])]
	json.dump(tmp2,f2)
	f2.write('\n')

f1.close()
f2.close()
