from tqdm import tqdm
import torch
import torch.nn as nn
from modelfile import Model, Config, EMA
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
from data_utils import Dataset, load_data, load_glove_model, load_emb_matrix
import os


data_dir = '../data'

elmo_emb_size = 256
elmo_options = data_dir + '/elmo/options.json'
elmo_weights = data_dir + '/elmo/weights.hdf5'


DELTA = 0.00000000001
weight_decay = 0.01

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


gamma_rank = 0.5
gamma_cls = 0.1
max_grad_norm = 1
ema_decay_rate = 0.99
init_normal_mean = 0
init_normal_std = 0.2


min_lr = 0
max_lr = 0.0002
lr_inc_steps = 5000
lr_multiply = (max_lr - min_lr)/lr_inc_steps
lr = min_lr
T_max = 75000

n_epochs = 50


### <PAD> token 	 -  [0,0,...,0,0]	... -> 300d glove
### <S> start token  -  [1,0,...,0,0]
### </S> end token   -  [0,1,...,0,0]

### QA Style Answer  -  [0,0,...,1,0]
### NLG Style Answer -  [0,0,...,0,1]


def train():
	
	steps = 1
	model.train()
	for epoch in tqdm(range(n_epochs)):

		train_loss = 0	
		train_loss_rank = 0
		train_loss_cls = 0
		train_loss_dec = 0
			
		for query_id, e_q, e_p, ch_q, ch_p, qseq_len, seq_len, q_words_idx, p_words_idx, p_labels, ans_pos, e_ans, ch_ans, aseq_len, ans_words_idx, ans_mask in tqdm(training_generator):
				
			if(steps <= lr_inc_steps):
				for param_group in optimizer.param_groups:
					param_group['lr'] = min_lr + lr_multiply * steps

			steps += 1

			inputs = {
					'e_q' : e_q.to(device),
					'e_p' : e_p.to(device),
					'ch_q' : ch_q.long().to(device),
					'ch_p' : ch_p.long().to(device),
					'qseq_len' : qseq_len.to(device),
					'q_words_idx' : q_words_idx.long().to(device),
					'seq_len' : seq_len.to(device),
					'p_words_idx' : p_words_idx.long().to(device),
					'e_ans' : e_ans.to(device),
					'ch_ans' : ch_ans.long().to(device),
					'aseq_len' : aseq_len.to(device)
			}

			
			beta_p, prob_ans, P_y = model(**inputs)

			# shape of beta_p - [batch_size, num_para]
			# shape of prob_ans - [batch_size]
			# shape of P_y - [batch_size, ans_len, ext_vocab_size]


			batch_size = p_labels.shape[0]
			
			p_labels = p_labels.to(device)
			ans_pos = ans_pos.to(device).view(batch_size)
			ans_words_idx = ans_words_idx.long().to(device)
			ans_mask = ans_mask.to(device)
			aseq_len = aseq_len.to(device)


			# loss for passage ranker

			l1_rank = p_labels * torch.log(beta_p + DELTA)
			l2_rank = (1 - p_labels) * torch.log(1 - beta_p + DELTA)
			l_rank = -torch.mean(l1_rank + l2_rank)

			# loss for answer possibility classifier

			l1_cls = ans_pos * torch.log(prob_ans + DELTA)
			l2_cls = (1 - ans_pos) * torch.log(1 - prob_ans + DELTA)
			l_cls = -torch.mean(l1_cls + l2_cls)


			# loss for decoder

			l_dec_tmp = P_y.gather(dim=2, index=ans_words_idx[:,:,None]).view(batch_size, max_ans_len)
			l_dec_tmp2 = torch.log(l_dec_tmp + DELTA)
			l_dec_tmp3 = torch.sum(ans_mask * l_dec_tmp2, dim=1)
			l_dec_tmp4 = (ans_pos * l_dec_tmp3)/aseq_len
			n_answerable = int(torch.sum(ans_pos))

			if(n_answerable > 0):
				l_dec = -torch.sum(l_dec_tmp4)/n_answerable
			else:
				l_dec = 0


			loss = l_dec + gamma_rank * l_rank + gamma_cls * l_cls

			train_loss += loss * batch_size
			train_loss_rank += l_rank * batch_size
			train_loss_cls += l_cls * batch_size
			train_loss_dec += l_dec * batch_size
			
			optimizer.zero_grad()
			loss.backward()

			clip_grad_norm_(model.parameters(), max_grad_norm)

			optimizer.step()


			# Apply Exponential Moving Average
			for name, param in model.named_parameters():
				if(param.requires_grad):
					param.data = ema(name, param.data)

			
			if(steps == lr_inc_steps):
				lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max)
			if(steps > lr_inc_steps):
				lr_scheduler.step()


		avg_loss = train_loss/len(train_data)
		avg_loss_rank = train_loss_rank/len(train_data)
		avg_loss_cls = train_loss_cls/len(train_data)
		avg_loss_dec = train_loss_dec/len(train_data)

		for param_group in optimizer.param_groups:
			cur_lr = param_group['lr']

		print('lr = ',cur_lr,', loss = ',avg_loss.item(),', loss rank = ', avg_loss_rank.item(),', loss cls = ', avg_loss_cls.item(),', loss dec = ', avg_loss_dec.item(), ', epoch = ',36+epoch+1)

		print('\n')
		
		torch.save({
			'model_state_dict': model.state_dict(),
		    'optimizer_state_dict': optimizer.state_dict(),
		}, 'checkpoints/saved_model_'+str(epoch)+'.pth')

	torch.save({
			'model_state_dict': model.state_dict(),
		    'optimizer_state_dict': optimizer.state_dict(),
		}, 'checkpoints/saved_model_'+str(epoch)+'.pth')



glove_word2vec = load_glove_model(data_dir, glove_vec_size)

emb_matrix, word2idx, idx2word = load_emb_matrix('/vocab.txt', glove_vec_size, glove_word2vec)
print('Done loading emb_matrix.')

train_data = load_data(data_dir + '/msmarco/train_v2.1.json', 35000, 15000)

print('Done loading Training data.')


train_params = {'batch_size': 32,
          'shuffle': True,
          'num_workers': 32,
	  	  'pin_memory': True}


training_set = Dataset(train_data, max_plen, max_qlen, max_ans_len, glove_vec_size, glove_word2vec, word2idx, qp_vocab_size, mode='train')
training_generator = DataLoader(training_set, **train_params)


cuda = torch.cuda.is_available()

device = torch.device('cpu')
if(cuda):
	device = torch.device('cuda')


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

# Initialize Model Parameters

for n,p in model.named_parameters(): 
	if('elmo' not in n): 
		if('bias' in n): 
			nn.init.zeros_(p) 
		else: 
			nn.init.normal_(p, init_normal_mean, init_normal_std) 


if(cuda):
	model = model.to(device)
	model = nn.DataParallel(model)


ema = EMA(ema_decay_rate)
for name, param in model.named_parameters():
	if param.requires_grad:
		ema.register(name, param.data)

optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)


if not os.path.exists('checkpoints'):
	os.makedirs('checkpoints')

train()
