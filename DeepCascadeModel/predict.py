from tqdm import tqdm
from nltk.tokenize import word_tokenize
import torch
from modelfile import Model, Config
import torch.optim as optim
import json
from torch.utils.data import DataLoader
from data_utils import Dataset, load_data
import os


data_dir = '../data'

elmo_emb_size = 256
elmo_options = data_dir + '/elmo/options.json'
elmo_weights = data_dir + '/elmo/weights.hdf5'

glove_vec_size = 300
hidden_size = 128

DELTA = 0.00000000001	# to avoid 0 in log
weight_decay = 0.0025	# L2 regularization term
max_plen = 128			# Max passage length
max_qlen = 32			# Max query length
num_para = 10			# Number of passages
lambda_de = 0.5			# multiplicative factor for Document Extraction loss
thres = 0.9				# threshold for Rouge-L

lr = 0.0003				# learning rate


def predict():

	model.eval()
	dev_loss = 0
	results_sel_para = []
	results_pred_start = []
	results_pred_end = []
	with torch.no_grad():
		for query_id, e_q, e_d, ch_q, ch_d, qseq_len, seq_len, p_labels, start, end, selected in tqdm(dev_generator):

			inputs = {
					'e_q' : e_q.to(device),
					'e_d' : e_d.to(device),
					'ch_q' : ch_q.long().to(device),
					'ch_d' : ch_d.long().to(device),
					'qseq_len' : qseq_len,
					'seq_len' : seq_len.to(device)
			}

			s_d_norm, start_alpha, end_alpha = model(**inputs)

			# shape of s_d_norm - [batch_size, num_para]
			# shape of start_alpha - [batch_size, num_para, max_plen]
			# shape of end_alpha - [batch_size, num_para, max_plen]


			batch_size = p_labels.shape[0]
			num_para = p_labels.shape[1]

			p_labels = p_labels.to(device)
			l1_de = p_labels * torch.log(s_d_norm + DELTA)
			l2_de = (1 - p_labels) * torch.log(1 - s_d_norm + DELTA)
			l_de = -torch.mean(torch.add(l1_de,l2_de))


			start = start.view(batch_size).to(device)
			end = end.view(batch_size).to(device)
			selected = selected.view(batch_size).to(device)

			alpha1_y1 = torch.diag(start_alpha[:, selected, start])
			st_loss = torch.log(alpha1_y1 + DELTA)

			alpha2_y2 = torch.diag(end_alpha[:, selected, end])
			en_loss = torch.log(alpha2_y2 + DELTA)
			l_ae = -torch.mean(torch.add(st_loss, en_loss))

			loss = torch.add(l_ae, lambda_de * l_de)

			dev_loss += loss*batch_size


			start_alpha_max_idx = torch.argmax(start_alpha, dim=2)
			end_alpha_max_idx = torch.argmax(end_alpha, dim=2)

			start_alpha_max = start_alpha.gather(dim=2, index=start_alpha_max_idx[:,:,None]).view(batch_size, num_para)
			end_alpha_max = end_alpha.gather(dim=2, index=end_alpha_max_idx[:,:,None]).view(batch_size, num_para)

			scores = s_d_norm * start_alpha_max * end_alpha_max

			selected_para = torch.argmax(scores, dim=1)

			pred_start = start_alpha_max_idx.gather(dim=1, index=selected_para[:,None]).view(batch_size)
			pred_end = end_alpha_max_idx.gather(dim=1, index=selected_para[:,None]).view(batch_size)

			results_sel_para.extend(selected_para.tolist())
			results_pred_start.extend(pred_start.tolist())
			results_pred_end.extend(pred_end.tolist())

		
		avg_loss = dev_loss/len(dev_data)
		print(avg_loss)

		return results_sel_para, results_pred_start, results_pred_end



dev_data = load_data('preprocessed_data/dev_data.json', thres, max_plen)[0:25000]

print('Done loading dev data.')

params = {'batch_size': 32,
          'shuffle': False,
          'num_workers': 32,
		  'pin_memory': True}


dev_set = Dataset(dev_data, max_plen, max_qlen, data_dir, glove_vec_size)
dev_generator = DataLoader(dev_set, **params)

config = Config(glove_vec_size, elmo_options, elmo_weights, elmo_emb_size, hidden_size, max_plen, max_qlen, num_para, device)
model = Model(config)
optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

if(cuda):
	model = model.to(device)

checkpoint = torch.load('checkpoints/saved_model.pth')

model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])


if not os.path.exists('outputs'):
	os.makedirs('outputs')


results_sel_para, results_pred_start, results_pred_end = predict()


f1 = open('outputs/dev_ref.json','w')
f2 = open('outputs/dev_pred.json','w')

for i in range(len(dev_data)):
	tmp1 = {}
	tmp1['query_id'] = dev_data[i]['query_id']
	tmp1['answers'] = dev_data[i]['answers']
	json.dump(tmp1,f1)
	f1.write('\n')

	tmp2 = {}
	tmp2['query_id'] = dev_data[i]['query_id']
	sel_para = dev_data[i]['passages'][results_sel_para[i]]['passage_text']
	pred_st = results_pred_start[i]
	pred_en = results_pred_end[i]
	ans_words = word_tokenize(sel_para)[pred_st:pred_en+1]
	tmp2['answers'] = [' '.join(ans_words)]
	json.dump(tmp2,f2)
	f2.write('\n')

f1.close()
f2.close()
