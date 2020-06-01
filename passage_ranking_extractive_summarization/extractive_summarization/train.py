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

max_s = 32			# Max number of sentences
max_slen = 64		# Max sentence length
max_qlen = 16		# Max query length
topk = 5			# Number of top passages to be selected

n_epochs = 100		# Number of epochs


def train(n_epochs):
	
	model.train()

	for epoch in tqdm(range(n_epochs)):

		train_loss = 0
		for query_id, e_q, e_s, qseq_len, n_sents, seq_len, labels, sent_mask in tqdm(training_generator):
			

			inputs = {
					'e_q' : e_q.to(device),
					'e_s' : e_s.to(device),
					'qseq_len' : qseq_len,
					'n_sents' : n_sents,
					'seq_len' : seq_len.to(device)
			}

			optimizer.zero_grad()

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

			train_loss += loss * batch_size
			loss.backward()
			optimizer.step()

			

		avg_loss = train_loss/len(data)
		print(avg_loss)
		
		torch.save({
			'model_state_dict': model.state_dict(),
		    'optimizer_state_dict': optimizer.state_dict(),
		}, 'checkpoints/saved_model_'+str(epoch)+'.pth')



	torch.save({
		'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
	}, 'checkpoints/saved_model.pth')




data = load_data('../passage_ranking/outputs/train_pred.json')

params = {'batch_size': 128,
          'shuffle': True,
          'num_workers': 20,
	  	  'pin_memory': True}


training_set = Dataset(data, max_s, max_slen, max_qlen, topk, glove_vec_size, data_dir)
training_generator = utils.data.DataLoader(training_set, **params)


device = torch.device('cpu')

cuda = torch.cuda.is_available()
if(cuda):
	device = torch.device('cuda')


config = Config(glove_vec_size, hidden_size, max_slen, max_qlen, max_s, device)

model = Model(config)
if(cuda):
	model = model.to(device)

optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

if not os.path.exists('checkpoints'):
	os.makedirs('checkpoints')

train()

