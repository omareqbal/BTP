from tqdm import tqdm
import torch
from modelfile import Model, Config
import json
import torch.optim as optim
from torch.utils.data import DataLoader
from data_utils import load_data, Dataset
import os


glove_vec_size = 300
hidden_size = 100

data_dir = '../../data'

lr = 0.002					# learning rate
DELTA = 0.0000000001		# to avoid 0 in log
weight_decay = 0.005		# L2 regularization term

max_s = 32			# Max number of sentences
max_slen = 64		# Max sentence length
max_qlen = 16		# Max query length
topk = 5			# Number of top passages to be selected

n_epochs = 100		# Number of epochs


def predict():

	model.eval()
	dev_loss = 0
	f1 = open('outputs/dev_ref.json','w')
	f2 = open('outputs/dev_pred.json','w')
	with torch.no_grad():
		for query_id, answers, sents, e_q, e_s, qseq_len, n_sents, seq_len, labels, sent_mask in tqdm(dev_generator):

			inputs = {
					'e_q' : e_q.to(device),
					'e_s' : e_s.to(device),
					'qseq_len' : qseq_len,
					'n_sents' : n_sents,
					'seq_len' : seq_len.to(device)
			}

			s_d_norm = model(**inputs)

				
			batch_size = labels.shape[0]

			labels = labels.to(device)
			sent_mask = sent_mask.to(device)
			l1 = labels * torch.log(s_d_norm + DELTA)
			l2 = (1 - labels) * torch.log(1 - s_d_norm + DELTA)
			l_de = torch.add(l1,l2)
			l_de2 = sent_mask * l_de
			total_sents = torch.sum(sent_mask)
			loss = -torch.sum(l_de2)/total_sents
			dev_loss += loss * batch_size


			for i in range(batch_size):
				tmp = {}
				tmp['query_id'] = int(query_id[i])
				tmp['answers'] = answers[i].split('#####')
				json.dump(tmp,f1)
				f1.write('\n')

				s_d_norm_ = s_d_norm[i] * sent_mask[i]
				temp = [{'score':s_d_norm_[j], 'index':j} for j in range(len(sents))]
				scores = sorted(temp, key=lambda x:x['score'], reverse=True)
				tmp2 = {}
				tmp2['query_id'] = int(query_id[i])
				tmp2['answers'] = [sents[scores[0]['index']][i]]
				json.dump(tmp2,f2)
				f2.write('\n')
		
		avg_loss = dev_loss/len(data)
		print(avg_loss)
		f1.close()
		f2.close()
		return invalid



data = load_data('../passage_ranking/outputs/dev_pred.json')

params = {'batch_size': 256,
          'shuffle': False,
          'num_workers': 16}


dev_set = Dataset(data, max_s, max_slen, max_qlen, topk)
dev_generator = utils.data.DataLoader(dev_set, **params)



device = torch.device('cpu')

cuda = torch.cuda.is_available()
if(cuda):
	device = torch.device('cuda')


config = Config(glove_vec_size, hidden_size, max_slen, max_qlen, max_s, device)

model = Model(config)
if(cuda):
	model = model.to(device)

optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)


checkpoint = torch.load('checkpoints/saved_trained_model.pth')

model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])


if not os.path.exists('outputs'):
	os.makedirs('outputs')

predict()