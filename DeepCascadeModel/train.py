from tqdm import tqdm
import torch
from modelfile import Model, Config
import torch.optim as optim
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

n_epochs = 50			# Number of epochs


def train():
	
	model.train()
	for epoch in tqdm(range(n_epochs)):

		train_loss = 0
		
		for query_id, e_q, e_d, ch_q, ch_d, qseq_len, seq_len, p_labels, start, end, selected in tqdm(training_generator):


			inputs = {
					'e_q' : e_q.to(device),
					'e_d' : e_d.to(device),
					'ch_q' : ch_q.long().to(device),
					'ch_d' : ch_d.long().to(device),
					'qseq_len' : qseq_len,
					'seq_len' : seq_len.to(device)
			}

			
			s_d_norm, start_alpha, end_alpha = model(**inputs)

			batch_size = p_labels.shape[0]
			
			p_labels = p_labels.to(device)
			l1_de = p_labels * torch.log(s_d_norm + DELTA)
			l2_de = (1 - p_labels) * torch.log(1 - s_d_norm + DELTA)
			l_de = -torch.mean(l1_de + l2_de)


			start = start.view(batch_size).to(device)
			end = end.view(batch_size).to(device)
			selected = selected.view(batch_size).to(device)

			alpha1_y1 = torch.diag(start_alpha[:, selected, start])
			st_loss = torch.log(alpha1_y1 + DELTA)

			alpha2_y2 = torch.diag(end_alpha[:, selected, end])
			en_loss = torch.log(alpha2_y2 + DELTA)
			l_ae = -torch.mean(st_loss + en_loss)

			loss = l_ae + lambda_de * l_de

			train_loss += loss * batch_size

			optimizer.zero_grad()
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



train_data = load_data('preprocessed_data/train_data.json', thres, max_plen)[0:100000]

print('Done loading Training data.')


train_params = {'batch_size': 32,
          'shuffle': True,
          'num_workers': 32,
	  	  'pin_memory': True}


training_set = Dataset(train_data, max_plen, max_qlen, data_dir, glove_vec_size)
training_generator = DataLoader(training_set, **train_params)

cuda = torch.cuda.is_available()

device = torch.device('cpu')
if(cuda):
	device = torch.device('cuda')

config = Config(glove_vec_size, elmo_options, elmo_weights, elmo_emb_size, hidden_size, max_plen, max_qlen, num_para, device)
model = Model(config)

if(cuda):
	model = model.to(device)

optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)


if not os.path.exists('checkpoints'):
	os.makedirs('checkpoints')

train()
