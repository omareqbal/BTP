from tqdm import tqdm
import torch
from modelfile import Model, Config
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

max_plen = 128		# Max passage length
max_qlen = 16		# Max query length
num_para = 10		# Number of passages

n_epochs = 100		# Number of epochs


def train(n_epochs):
	
	model.train()
	for epoch in tqdm(range(n_epochs)):

		train_loss = 0
		
		for query_id, e_q, e_d, qseq_len, seq_len, labels in tqdm(training_generator):
			
			inputs = {
					'e_q' : e_q.to(device),
					'e_d' : e_d.to(device),
					'qseq_len' : qseq_len,
					'seq_len' : seq_len.to(device)
			}

			optimizer.zero_grad()
			s_d_norm = model(**inputs)	
			# shape of s_d_norm - [batch_size, num_para]

			batch_size = labels.shape[0]
			
			labels = labels.to(device)

			# computing loss
			l1 = labels * torch.log(s_d_norm + DELTA)
			l2 = (1 - labels) * torch.log(1 - s_d_norm + DELTA)
			l_de = torch.add(l1,l2)
			loss = -torch.mean(l_de)


			train_loss += loss * batch_size
			loss.backward()
			optimizer.step()
		

		avg_loss = train_loss/len(train_data)

		print('training loss = ',avg_loss.item(),' epoch = ',epoch+1)
		
		torch.save({
			'model_state_dict': model.state_dict(),
		    'optimizer_state_dict': optimizer.state_dict(),
		}, 'checkpoints/saved_model_'+str(epoch)+'.pth')


	torch.save({
			'model_state_dict': model.state_dict(),
		    'optimizer_state_dict': optimizer.state_dict(),
		}, 'checkpoints/saved_model.pth')



train_data = load_data(data_dir + '/msmarco/train_v2.1.json')

print('Done loading Training data.')


params = {'batch_size': 128,
          'shuffle': True,
          'num_workers': 20,
	  	  'pin_memory': True}


training_set = Dataset(data, max_plen, max_qlen, glove_vec_size, data_dir)
training_generator = DataLoader(training_set, **params)


device = torch.device('cpu')

cuda = torch.cuda.is_available()
if(cuda):
	device = torch.device('cuda')


config = Config(glove_vec_size, hidden_size, max_plen, max_qlen, num_para, device)

model = Model(config)
if(cuda):
	model = model.to(device)

optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

if not os.path.exists('checkpoints'):
	os.makedirs('checkpoints')

train()