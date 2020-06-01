import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

# Model Configuration

class Config(object):
	def __init__(self,
				emb_size,
				hidden_size,
				max_slen,
				max_qlen,
				max_s,
				device):
		self.emb_size = emb_size
		self.hidden_size = hidden_size
		self.max_slen = max_slen
		self.max_qlen = max_qlen
		self.max_s = max_s
		self.device = device



class Model(nn.Module):
	def __init__(self, config):
		super(Model, self).__init__()
		self.config = config

		self.bilstm1 = nn.LSTM(input_size=config.emb_size, hidden_size=config.hidden_size, bidirectional=True, batch_first=True)
		self.bilstm2 = nn.LSTM(input_size=config.emb_size, hidden_size=config.hidden_size, bidirectional=True, batch_first=True)

		self.linear1 = nn.Linear(2*config.hidden_size, 2*config.hidden_size)
		self.linear2 = nn.Linear(2*config.hidden_size, 2*config.hidden_size)
		self.linear3 = nn.Linear(2*config.hidden_size, 2*config.hidden_size)

		self.fuse_linear1 = nn.Linear(4*config.hidden_size, 2*config.hidden_size)
		self.fuse_linear2 = nn.Linear(4*config.hidden_size, 2*config.hidden_size)

		self.bilinear = nn.Bilinear(2*config.hidden_size, 2*config.hidden_size, 1)
		self.align_linear1 = nn.Linear(2*config.hidden_size, 1)
		self.align_linear2 = nn.Linear(2*config.hidden_size, 1)


	def forward(self, e_q, e_s, qseq_len, n_sents, seq_len):
	 	# e : word embedding
	 	# e_q - shape - [batch_size, q_len, emb_size]
	 	# e_d - shape - [batch_size, max_s, s_len, emb_size]


	 	#
	 	#	SHARED Q&D MODELLING
	 	#

	 	batch_size = e_q.shape[0]
	 	emb_size = self.config.emb_size
	 	max_s = self.config.max_s
	 	s_len = self.config.max_slen
	 	q_len = self.config.max_qlen
	 	hidden_size = self.config.hidden_size
	 	device = self.config.device

	 	
	 	# shape of e_q - [batch_size, q_len, emb_size]

	 	emb_q = pack_padded_sequence(e_q, qseq_len, batch_first=True, enforce_sorted=False)
	 	u_q, _ = self.bilstm1(emb_q)	# shape of u_q - [batch_size, q_len, 2*hidden_size]
	 	u_q, _ = pad_packed_sequence(u_q, batch_first=True)

	 	q_len = u_q.shape[1]
		
	 	e_s = e_s.view(batch_size*max_s, s_len, emb_size)
	 	seq_len = seq_len.view(batch_size*max_s)

	 	emb_d = pack_padded_sequence(e_s, seq_len, batch_first=True, enforce_sorted=False)
	 	u_d, _ = self.bilstm2(emb_d)
	 	u_d, _ = pad_packed_sequence(u_d, batch_first=True)

	 	# shape of u_d - [batch_size*max_s, s_len, 2*hidden_size]

	 	s_len = u_d.shape[1]

	 	# shape of u_d - [batch_size, max_s, s_len, 2*hidden_size]
	 	u_d = u_d.view(batch_size, max_s, s_len, 2*hidden_size)
	 	seq_len = seq_len.view(batch_size, max_s)

	 	#
	 	# Co-attention and Fusion
	 	#

	 	# shape of u_q_temp - [batch_size, q_len, 2*hidden_size]
	 	u_q_temp = F.relu(self.linear1(u_q))
	 	
	 	# shape of u_d_temp - [batch_size, max_s, s_len, 2*hidden_size]
	 	u_d_temp = F.relu(self.linear2(u_d))

	 	u_q_temp2 = u_q_temp.transpose(1,2) # shape - [batch_size, 2*hidden_size, q_len]

	 	s = torch.matmul(u_d_temp, u_q_temp2.view(batch_size, 1, 2*hidden_size, q_len))
	 	
	 	alpha = F.softmax(s, dim=3) # shape -[batch_size, max_s, s_len, q_len]

	 	u_d_att = torch.matmul(alpha, u_q_temp.view(batch_size, 1, q_len, 2*hidden_size))

	 	#shape - [batch_size, max_s, s_len, 2*hidden_size]
	 	
	 	v_d = self.fuse_linear1(torch.cat([u_d, u_d_att], dim=3)) #shape - [batch_size, max_s, s_len, 2*hidden_size]

	 	#
	 	# Self-attention and Fusion
	 	#

	 	v_d_temp = self.linear3(v_d)

	 	s = torch.matmul(v_d_temp, v_d_temp.transpose(2,3))
	 	
	 	beta = F.softmax(s, dim=3) # shape - [batch_size, max_s, s_len, s_len]

	 		 		
	 	v_d_att = torch.matmul(beta, v_d)

	 	d_d = self.fuse_linear2(torch.cat([v_d, v_d_att], dim=3))  #shape - [batch_size, max_s, s_len, 2*hidden_size]

	 	#
	 	# Self-align for query
	 	#

	 	s = self.align_linear1(u_q).view(batch_size, q_len)

	 	gamma = F.softmax(s, dim=1)

	 	r_q = torch.matmul(gamma.view(batch_size, 1, q_len), u_q)
	 	r_q = r_q.view(batch_size, 2*hidden_size)

	 	#
	 	#	SENTENCE RANKING
	 	#

	 	#shape of d_d - [batch_size, max_s, s_len, 2*hidden_size]
	 	s = self.align_linear2(d_d).view(batch_size, max_s, s_len)
	 	mu = F.softmax(s, dim=2)
	 	

	 	r_d = torch.matmul(mu.view(batch_size, max_s, 1, s_len), d_d)
	 	r_d = r_d.view(batch_size, max_s, 2*hidden_size)
	 	
	 	
	 	r_d2 = r_d.transpose(0,1)
	 	s_d = torch.zeros(max_s, batch_size, device=device)
	 	for n in range(max_s):
	 		s_d[n] = self.bilinear(r_q, r_d2[n]).view(batch_size)

	 	s_d_norm = torch.sigmoid(s_d)

	 	s_d_norm2 = s_d_norm.transpose(0,1)

	 	return s_d_norm2
