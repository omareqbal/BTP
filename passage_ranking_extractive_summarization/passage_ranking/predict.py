from tqdm import tqdm
import json
import torch
from modelfile import Model, Config
import torch.optim as optim
from torch.utils.data import DataLoader
from data_utils import load_data, Dataset
import os
import argparse


glove_vec_size = 300
hidden_size = 100

data_dir = '../../data'

lr = 0.002					# learning rate
DELTA = 0.0000000001		# to avoid 0 in log
weight_decay = 0.005		# L2 regularization term

max_plen = 128		# Max passage length
max_qlen = 16		# Max query length
num_para = 10		# Number of passages

n_epochs = 100		# Number of epochs

parser = argparse.ArgumentParser()
parser.add_argument('--data', help='dataset to run the prediction on - train | dev')

args = parser.parse_args()


def predict():

	model.eval()
	dev_loss = 0
	f = open('outputs/' + args.data +'_pred.json','w')
	with torch.no_grad():
		for row, e_q, e_d, qseq_len, seq_len, labels in tqdm(dev_generator):
			inputs = {
					'e_q' : e_q.to(device),
					'e_d' : e_d.to(device),
					'qseq_len' : qseq_len,
					'seq_len' : seq_len.to(device)
			}

			s_d_norm = model(**inputs)
			
			batch_size = labels.shape[0]
			labels = labels.to(device)

			# computing loss
			l1 = labels * torch.log(s_d_norm + DELTA)
			l2 = (1 - labels) * torch.log(1 - s_d_norm + DELTA)
			l_de = torch.add(l1,l2)
			loss = -torch.mean(l_de)

			dev_loss += loss * batch_size

			for i in range(batch_size):
				tmp = {}
				tmp['query_id'] = int(row['query_id'][i])
				tmp['query'] = row['query'][i]
				tmp['answers'] = row['answers'][i].split('#####')
				tmp['passages'] = []
				for n,p in enumerate(row['passages']):
					tmp2 = {}
					tmp2['passage_text'] = p['passage_text'][i]
					tmp2['is_selected'] = int(p['is_selected'][i])
					tmp2['score'] = float(s_d_norm[i][n])
					tmp['passages'].append(tmp2)

				json.dump(tmp,f)
				f.write('\n')
		
		avg_loss = dev_loss/len(data)
		print(avg_loss)
		f.close()
		return invalid



data = load_data(data_dir + '/msmarco/' + args.data +'_v2.1.json')

params = {'batch_size': 256,
          'shuffle': False,
          'num_workers': 16}

dev_set = Dataset(data, max_plen, max_qlen, glove_vec_size, data_dir)
dev_generator = utils.data.DataLoader(dev_set, **params)

device = torch.device('cpu')

cuda = torch.cuda.is_available()
if(cuda):
	device = torch.device('cuda')

config = Config(glove_vec_size, hidden_size, max_plen, max_qlen, num_para, device)


model = Model(config)
optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

if(cuda):
	model = model.to(device)

checkpoint = torch.load('checkpoints/saved_model.pth')

model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

if not os.path.exists('outputs'):
	os.makedirs('outputs')

predict()