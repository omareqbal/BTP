import torch
import torch.nn as nn
import torch.nn.functional as F
from allennlp.modules.elmo import Elmo


########## Model Configuration ###############

class Config(object):
	def __init__(self,
				glove_emb_size,
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
				device):
		
		self.glove_emb_size = glove_emb_size
		self.elmo_options = elmo_options
		self.elmo_weights = elmo_weights
		self.elmo_emb_size = elmo_emb_size
		self.common_vocab_size = common_vocab_size
		self.qp_vocab_size = qp_vocab_size
		self.hidden_size = hidden_size
		self.emb_size = glove_emb_size + elmo_emb_size + 4
		self.max_plen = max_plen
		self.max_qlen = max_qlen
		self.max_ans_len = max_ans_len
		self.num_para = num_para
		self.n_attention_heads = n_attention_heads
		self.dropout = dropout
		self.dim_feedforward = dim_feedforward
		self.n_shared_encoder_blocks = n_shared_encoder_blocks
		self.n_modeling_encoder_blocks_q = n_modeling_encoder_blocks_q
		self.n_modeling_encoder_blocks_p = n_modeling_encoder_blocks_p
		self.n_decoder_blocks = n_decoder_blocks
		self.device = device


####### Exponential Moving Average ############

class EMA():
	def __init__(self, mu):
		self.mu = mu
		self.shadow = {}

	def register(self, name, val):
		self.shadow[name] = val.clone()

	def __call__(self, name, x):
		assert name in self.shadow
		new_average = (1.0 - self.mu) * x + self.mu * self.shadow[name]
		self.shadow[name] = new_average.clone()
		return new_average


############ Decoder Block ###############

class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=256, dropout=0.3, activation='relu'):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn_q = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn_p = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.norm4 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.dropout4 = nn.Dropout(dropout)

        if(activation == 'gelu'):
        	self.activation = F.gelu
        else:
        	self.activation = F.relu

    
    def forward(self, tgt, memory_q , memory_p, tgt_mask=None, memory_q_mask=None, memory_p_mask=None,
                tgt_key_padding_mask=None, memory_q_key_padding_mask=None,  memory_p_key_padding_mask=None):

        tgt2 = self.self_attn(tgt, tgt, tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2 = self.multihead_attn_q(tgt, memory_q, memory_q, attn_mask=memory_q_mask,
                                   key_padding_mask=memory_q_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.multihead_attn_p(tgt, memory_p, memory_p, attn_mask=memory_p_mask,
                                   key_padding_mask=memory_p_key_padding_mask)[0]
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))

        tgt = tgt + self.dropout4(tgt2)
        tgt = self.norm4(tgt)
        return tgt




################### QUESTION PASSAGES READER #################
##################### PASSAGE RANKER #########################
################ ANSWER POSSIBILITY CLASSIFIER ###############


class ReaderRankerClassifier(nn.Module):
	def __init__(self, config, elmo):
		super().__init__()
		self.config = config
		
		self.elmo = elmo

		self.highway_transform1 = nn.Linear(config.emb_size, config.emb_size)
		self.highway_gate1 = nn.Linear(config.emb_size, config.emb_size)
		self.highway_transform2 = nn.Linear(config.emb_size, config.emb_size)
		self.highway_gate2 = nn.Linear(config.emb_size, config.emb_size)

		self.shared_encoder_linear = nn.Linear(config.emb_size, config.hidden_size)

		transformer_shared_encoder_layer = nn.TransformerEncoderLayer(config.hidden_size, config.n_attention_heads, config.dim_feedforward, config.dropout, activation='gelu')
		self.transformer_shared_encoder = nn.TransformerEncoder(transformer_shared_encoder_layer, config.n_shared_encoder_blocks)

		self.dual_attention_linear = nn.Linear(3 * config.hidden_size, 1, bias=False)

		self.modeling_linear_q = nn.Linear(5 * config.hidden_size, config.hidden_size)
		self.modeling_linear_p = nn.Linear(5 * config.hidden_size, config.hidden_size)

		transformer_modeling_encoder_layer_q = nn.TransformerEncoderLayer(config.hidden_size, config.n_attention_heads, config.dim_feedforward, config.dropout, activation='gelu')
		self.transfomer_modeling_encoder_q = nn.TransformerEncoder(transformer_modeling_encoder_layer_q, config.n_modeling_encoder_blocks_q)

		transformer_modeling_encoder_layer_p = nn.TransformerEncoderLayer(config.hidden_size, config.n_attention_heads, config.dim_feedforward, config.dropout, activation='gelu')
		self.transfomer_modeling_encoder_p = nn.TransformerEncoder(transformer_modeling_encoder_layer_p, config.n_modeling_encoder_blocks_p)
		
		
		self.ranker_linear = nn.Linear(config.hidden_size, 1, bias=False)

		self.ans_pos_linear = nn.Linear(config.num_para * config.hidden_size, 1, bias=False)



	def forward(self, e_q, e_p, ch_q, ch_p, qseq_len, seq_len):
		# e : word embedding
		# c : character embedding
		# x_q - shape - [batch_size, q_len, emb_size]
		# x_p - shape - [batch_size, num_para, p_len, emb_size]
		# qseq_len - shape - [batch_size]
		# seq_len - shape - [batch_size, num_para]


		batch_size = e_q.shape[0]
		emb_size = self.config.emb_size
		p_len = self.config.max_plen
		q_len = self.config.max_qlen
		num_para = self.config.num_para
		hidden_size = self.config.hidden_size


		c_q = self.elmo(ch_q)['elmo_representations'][0]

		ch_p = ch_p.view(batch_size * num_para, p_len, -1)
		
		c_p = self.elmo(ch_p)['elmo_representations'][0]
		c_p = c_p.view(batch_size, num_para, p_len, -1)

		x_q = torch.cat([e_q, c_q], dim=2)
		x_p = torch.cat([e_p, c_p], dim=3)


		############## QUESTION PASSAGES READER #################
		
		#
		# Word Embedding Layer
		#


		# 2 layer highway network on query

		hq_out = self.highway_network(x_q)
		# shape of hq_out - [batch_size, q_len, emb_size]


		# 2 layer highway network on passages

		x_p = x_p.view(batch_size * num_para, p_len, emb_size)
		seq_len = seq_len.view(batch_size * num_para)

		hp_out = self.highway_network(x_p)
		# shape of hp_out - [batch_size * num_para, p_len, emb_size]


		#
		# Shared Encoder Layer
		#

		Eq = self.shared_encoder_layer(hq_out, qseq_len, q_len)  # shape - [batch_size, q_len, hidden_size]

		Ep = self.shared_encoder_layer(hp_out, seq_len, p_len)	# shape - [batch_size * num_para, p_len, hidden_size]

		Ep = Ep.view(batch_size, num_para, p_len, hidden_size)


		#
		# Dual Attention Layer
		#

		G_q_p_k, G_p_q = self.dual_attention_layer(Eq, Ep)
		
		# shape of G_q_p_k - [batch_size, num_para, 5 * hidden_size, p_len]
		# shape of G_p_q - [batch_size, 5 * hidden_size, q_len]


		#
		# Modeling Encoder Layer
		#

		M_p_k, p_mask = self.modeling_encoder_layer_p(G_q_p_k.view(batch_size * num_para, 5 * hidden_size, p_len), seq_len)
		M_p_k = M_p_k.contiguous().view(batch_size, num_para, p_len, hidden_size)
		p_mask = p_mask.view(batch_size, num_para, p_len)

		M_q, q_mask = self.modeling_encoder_layer_q(G_p_q, qseq_len)
		# shape - [batch_size, q_len, hidden_size]




		################## PASSAGE RANKER ########################
		M_1_p_k = M_p_k[:, :, 0, :].contiguous()  # shape - [batch_size, num_para, hidden_size]
		
		beta_p = torch.sigmoid(self.ranker_linear(M_1_p_k).view(batch_size, num_para))
		
		# shape - [batch_size, num_para]		



		################ ANSWER POSSIBILITY CLASSIFIER #################
		M_1_p_k_all = M_1_p_k.view(batch_size, num_para * hidden_size)

		prob_ans = torch.sigmoid(self.ans_pos_linear(M_1_p_k_all).view(batch_size))

		# shape - [batch_size]


		return beta_p, prob_ans, M_q, q_mask, M_p_k, p_mask


	def highway_network(self, x):
		
		h_transform1 = F.relu(self.highway_transform1(x))
		h_gate1 = torch.sigmoid(self.highway_gate1(x))
		h_out1 = h_gate1 * h_transform1 + (1 - h_gate1) * x

		h_transform2 = F.relu(self.highway_transform2(h_out1))
		h_gate2 = torch.sigmoid(self.highway_gate2(h_out1))
		h_out2 = h_gate2 * h_transform2 + (1 - h_gate2) * h_out1

		return h_out2


	def shared_encoder_layer(self, src, seq_len, max_len):

		batch_size = seq_len.shape[0]

		mask = torch.arange(max_len, device=self.config.device).expand(batch_size, max_len)>= seq_len.unsqueeze(1) 

		src_temp = self.shared_encoder_linear(src)

		out = self.transformer_shared_encoder(src_temp.transpose(0,1), src_key_padding_mask=mask)

		return out.transpose(0,1)


	def dual_attention_layer(self, Eq, Ep):

		batch_size = Eq.shape[0]
		hidden_size = self.config.hidden_size
		p_len = self.config.max_plen
		q_len = self.config.max_qlen
		num_para = self.config.num_para

		
		Eq_p1 = Ep[:,:,:,None,:] * Eq[:,None,None,:,:] 
		# shape - [batch_size, num_para, p_len, q_len, hidden_size]

		Eq_temp = Eq.view(batch_size, 1, q_len, hidden_size).expand(batch_size, num_para, q_len, hidden_size)
		Eq_temp2 = Eq_temp.view(batch_size, num_para, 1, q_len, hidden_size).expand(batch_size, num_para, p_len, q_len, hidden_size)
		Ep_temp2 = Ep.view(batch_size, num_para, p_len, 1, hidden_size).expand(batch_size, num_para, p_len, q_len, hidden_size)

		Eq_p2 = torch.cat([Eq_temp2, Eq_p1], dim=4)
		Eq_p = torch.cat([Ep_temp2, Eq_p2], dim=4)
		# shape - [batch_size, num_para, p_len, q_len, 3 * hidden_size]


		U_p_k = self.dual_attention_linear(Eq_p).view(batch_size, num_para, p_len, q_len)

		A_p_k = (F.softmax(U_p_k, dim=3)).transpose(2,3) # shape - [batch_size, num_para, q_len, p_len]
		B_p_k = F.softmax(U_p_k, dim=2)				   # shape - [batch_size, num_para, p_len, q_len]

		Eq_temp3 = Eq_temp.transpose(2,3)  # shape - [batch_size, num_para, hidden_size, q_len]

		A_p_k1 = torch.matmul(Eq_temp3, A_p_k) # shape - [batch_size, num_para, hidden_size, p_len]


		Ep_temp3 = Ep.transpose(2,3) # shape - [batch_size, num_para, hidden_size, p_len]

		B_p_k1 = torch.matmul(Ep_temp3, B_p_k) # shape - [batch_size, num_para, hidden_size, q_len]

		A_p_k2 = torch.matmul(B_p_k1, A_p_k)   # shape - [batch_size, num_para, hidden_size, p_len]

		B_p_k2 = torch.matmul(A_p_k1, B_p_k)   # shape - [batch_size, num_para, hidden_size, q_len]

		E_p_k_A_p_k1 = Ep_temp3 * A_p_k1  # shape - [batch_size, num_para, hidden_size, p_len]
		E_p_k_A_p_k2 = Ep_temp3 * A_p_k2

		Eq_temp4 = Eq.transpose(1,2)    # shape - [batch_size, hidden_size, q_len]
		B1, _ = torch.max(B_p_k1, dim=1)  # shape - [batch_size, hidden_size, q_len]
		B2, _ = torch.max(B_p_k2, dim=1)

		E_q_B1 = Eq_temp4 * B1  # shape - [batch_size, hidden_size, q_len]
		E_q_B2 = Eq_temp4 * B2

		G_q_p_k = torch.cat([Ep_temp3, A_p_k1, A_p_k2, E_p_k_A_p_k1, E_p_k_A_p_k2], dim=2)
		# shape - [batch_size, num_para, 5 * hidden_size, p_len]


		G_p_q = torch.cat([Eq_temp4, B1, B2, E_q_B1, E_q_B2], dim=1)
		# shape - [batch_size, 5 * hidden_size, q_len]


		return G_q_p_k, G_p_q


	def modeling_encoder_layer_p(self, src, seq_len):

		# shape of src - [batch_size * num_para, 5 * hidden_size, p_len]
		batch_size = seq_len.shape[0]
		num_para = self.config.num_para
		p_len = self.config.max_plen

		mask = torch.arange(p_len, device=self.config.device).expand(batch_size, p_len)>= seq_len.unsqueeze(1) 

		src_temp = self.modeling_linear_p(src.transpose(1,2)) # shape - [batch_size * num_para, p_len, hidden_size]

		out = self.transfomer_modeling_encoder_p(src_temp.transpose(0,1), src_key_padding_mask=mask)

		return out.transpose(0,1), mask


	def modeling_encoder_layer_q(self, src, seq_len):

		# shape of src - [batch_size, 5 * hidden_size, q_len]
		batch_size = seq_len.shape[0]
		q_len = self.config.max_qlen

		mask = torch.arange(q_len, device=self.config.device).expand(batch_size, q_len)>= seq_len.unsqueeze(1) 

		src_temp = self.modeling_linear_q(src.transpose(1,2)) # shape - [batch_size, q_len, hidden_size]

		out = self.transfomer_modeling_encoder_q(src_temp.transpose(0,1), src_key_padding_mask=mask)

		return out.transpose(0,1), mask




##################### ANSWER SENTENCE DECODER ######################

class AnswerDecoder(nn.Module):
	def __init__(self, config, emb_matrix, elmo):
		super().__init__()
		self.config = config
		
		self.emb_matrix_ = emb_matrix   # shape - [common_vocab_size, glove_emb_size+2]
		self.register_buffer('emb_matrix', self.emb_matrix_)

		self.elmo = elmo

		self.highway_transform_dec1 = nn.Linear(config.emb_size, config.emb_size)
		self.highway_gate_dec1 = nn.Linear(config.emb_size, config.emb_size)
		self.highway_transform_dec2 = nn.Linear(config.emb_size, config.emb_size)
		self.highway_gate_dec2 = nn.Linear(config.emb_size, config.emb_size)


		self.transformer_decoder_linear = nn.Linear(config.emb_size, config.hidden_size)
		self.transformer_decoder_layers = nn.ModuleList([TransformerDecoderLayer(config.hidden_size, config.n_attention_heads, config.dim_feedforward, config.dropout, activation='gelu') for i in range(config.n_decoder_blocks)])

		self.ext_v_dist_linear = nn.Linear(config.hidden_size, config.glove_emb_size+2)

		self.copy_dist_linear_pm = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
		self.copy_dist_linear_ps = nn.Linear(config.hidden_size, config.hidden_size)
		self.copy_dist_linear_p = nn.Linear(config.hidden_size, 1, bias=False)

		self.copy_dist_linear_qm = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
		self.copy_dist_linear_qs = nn.Linear(config.hidden_size, config.hidden_size)
		self.copy_dist_linear_q = nn.Linear(config.hidden_size, 1, bias=False)

		self.final_dist_linear = nn.Linear(3 * config.hidden_size, 3)


	def forward(self, e_ans, ch_ans, aseq_len, M_q, q_mask, M_p_k, p_mask, beta_p, q_words_idx, p_words_idx):
		# e : word embedding
		# c : character embedding
		# x_ans - shape - [batch_size, ans_len, emb_size]
		# aseq_len - shape - [batch_size]
		# shape of M_q - [batch_size, q_len, hidden_size]
		# shape of q_mask- [batch_size, q_len]
		# shape of M_p_k - [batch_size, num_para, p_len, hidden_size]
		# shape of p_mask- [batch_size, num_para, p_len]
		# shape of beta_p - [batch_size, num_para]
		# shape of q_words_idx - [batch_size, q_len]
		# shape of p_words_idx - [batch_size, num_para, p_len]
		

		batch_size = e_ans.shape[0]
		emb_size = self.config.emb_size
		p_len = self.config.max_plen
		q_len = self.config.max_qlen
		ans_len = self.config.max_ans_len
		num_para = self.config.num_para
		hidden_size = self.config.hidden_size


		#
		# Word Embedding Layer
		#

		c_ans = self.elmo(ch_ans)['elmo_representations'][0]

		x_ans = torch.cat([e_ans, c_ans], dim=2) # shape - [batch_size, ans_len, emb_size]


		# 2 layer highway network on answer
		ha_out = self.highway_network_dec(x_ans) # shape - [batch_size, ans_len, emb_size]


		#
		# Attentional Decoder Layer
		#

		
		p_mask = p_mask.view(batch_size, num_para * p_len)

		s = self.attentional_decoder_layer(ha_out, aseq_len, M_q, q_mask, M_p_k, p_mask)

		# shape of s - [batch_size, ans_len, hidden_size]


		#
		# Multi-source Pointer-Generator
		#

		P_y = self.multi_src_pointer_gen(M_q, M_p_k, s, beta_p, q_words_idx, p_words_idx)

		# shape of P_y - [batch_size, ans_len, ext_vocab_size]

		return P_y


	def highway_network_dec(self, x):
		
		h_transform1 = F.relu(self.highway_transform_dec1(x))
		h_gate1 = torch.sigmoid(self.highway_gate_dec1(x))
		h_out1 = h_gate1 * h_transform1 + (1 - h_gate1) * x

		h_transform2 = F.relu(self.highway_transform_dec2(h_out1))
		h_gate2 = torch.sigmoid(self.highway_gate_dec2(h_out1))
		h_out2 = h_gate2 * h_transform2 + (1 - h_gate2) * h_out1

		return h_out2


	def attentional_decoder_layer(self, ans, aseq_len, M_q, q_mask, M_p_k, p_mask):

		batch_size = aseq_len.shape[0]
		p_len = self.config.max_plen
		q_len = self.config.max_qlen
		ans_len = self.config.max_ans_len
		num_para = self.config.num_para
		hidden_size = self.config.hidden_size
		device = self.config.device

		M_p_all = M_p_k.view(batch_size, num_para * p_len, hidden_size)

		ans_mask = torch.arange(ans_len, device=device).expand(batch_size, ans_len)>= aseq_len.unsqueeze(1) 

		mask = (torch.triu(torch.ones(ans_len, ans_len, device=device)) == 1).transpose(0, 1)
		ans_seq_mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))


		out = self.transformer_decoder_linear(ans.transpose(0,1))

		for i in range(self.config.n_decoder_blocks):
			out = self.transformer_decoder_layers[i](out,
						M_q.transpose(0,1), M_p_all.transpose(0,1), tgt_mask=ans_seq_mask, tgt_key_padding_mask=ans_mask,
						memory_q_key_padding_mask=q_mask, memory_p_key_padding_mask=p_mask)

		return out.transpose(0,1)


	def multi_src_pointer_gen(self, M_q, M_p_k, s, beta_p, q_words_idx, p_words_idx):
		# shape of s - [batch_size, ans_len, hidden_size]
		batch_size = s.shape[0]
		p_len = self.config.max_plen
		q_len = self.config.max_qlen
		num_para = self.config.num_para
		ans_len = self.config.max_ans_len
		hidden_size = self.config.hidden_size
		qp_vocab_size = self.config.qp_vocab_size
		device = self.config.device

		ext_vocab_size = self.config.common_vocab_size + qp_vocab_size


		#
		# Extended Vocabulary Distribution
		#

		P_v_temp = self.ext_v_dist_linear(s) # shape - [batch_size, ans_len, glove_emb_size+2]

		# shape of emb_matrix - [common_vocab_size, glove_emb_size+2]

		P_v_temp2 = torch.matmul(P_v_temp, self.emb_matrix.transpose(0,1))  # shape - [batch_size, ans_len, common_vocab_size]
		P_v_temp3 = F.softmax(P_v_temp2, dim=2)

		P_v = torch.cat([P_v_temp3, torch.zeros(batch_size, ans_len, qp_vocab_size, device=device)], dim=2)	# shape - [batch_size, ans_len, ext_vocab_size]


		#
		# Copy Distribution for Passages
		#

		e_pk_temp = self.copy_dist_linear_pm(M_p_k) # shape - [batch_size, num_para, p_len, hidden_size]
		e_pk_temp2 = self.copy_dist_linear_ps(s) # shape - [batch_size, ans_len, hidden_size]

		e_pk_temp3 = e_pk_temp[:,None,:,:,:] + e_pk_temp2[:,:,None,None,:]
		# shape - [batch_size, ans_len, num_para, p_len, hidden_size]

		e_pk = self.copy_dist_linear_p(e_pk_temp3).view(batch_size, ans_len, num_para, p_len)

		alpha_p_temp = F.softmax(e_pk.view(batch_size, ans_len, num_para * p_len), dim=2)

		# Combined Attentions

		alpha_p_temp2 = alpha_p_temp.view(batch_size, ans_len, num_para, p_len)
		alpha_p_temp3 = (alpha_p_temp2 * beta_p[:,None,:,None]).view(batch_size, ans_len, num_para * p_len)

		alpha_p = alpha_p_temp3/torch.sum(alpha_p_temp3, dim=2)[:,:,None]


		p_words_idx_all = p_words_idx.view(batch_size, num_para * p_len)

		P_p = torch.zeros(batch_size, ans_len, ext_vocab_size, device=device)

		P_p.scatter_add_(dim=2, index=p_words_idx_all[:,None,:].expand(batch_size, ans_len, num_para * p_len), src=alpha_p)

		# shape of P_p - [batch_size, ans_len, ext_vocab_size]


		M_p_all = M_p_k.view(batch_size, num_para * p_len, hidden_size)

		c_p_temp = alpha_p[:,:,:,None] * M_p_all[:,None,:,:] 
		# shape - [batch_size, ans_len, num_para * p_len, hidden_size]

		c_p = torch.sum(c_p_temp, dim=2) # shape - [batch_size, ans_len, hidden_size]

		
		#
		# Copy Distribution for Query
		#

		e_q_temp = self.copy_dist_linear_qm(M_q) # shape - [batch_size, q_len, hidden_size]
		e_q_temp2 = self.copy_dist_linear_qs(s) # shape - [batch_size, ans_len, hidden_size]

		e_q_temp3 = e_q_temp[:,None,:,:] + e_q_temp2[:,:,None,:]
		# shape - [batch_size, ans_len, q_len, hidden_size]

		e_q = self.copy_dist_linear_q(e_q_temp3).view(batch_size, ans_len, q_len)

		alpha_q = F.softmax(e_q, dim=2)

		P_q = torch.zeros(batch_size, ans_len, ext_vocab_size, device=device)
		
		P_q.scatter_add_(dim=2, index=q_words_idx[:,None,:].expand(batch_size, ans_len, q_len), src=alpha_q)

		# shape of P_q - [batch_size, ans_len, ext_vocab_size]



		c_q_temp = alpha_q[:,:,:,None] * M_q[:,None,:,:] 
		# shape - [batch_size, ans_len, q_len, hidden_size]

		c_q = torch.sum(c_q_temp, dim=2) # shape - [batch_size, ans_len, hidden_size]

		lambda_vqp_temp = self.final_dist_linear(torch.cat([s,c_q,c_p], dim=2)) # shape - [batch_size, ans_len, 3]

		lambda_vqp = F.softmax(lambda_vqp_temp, dim=2)

		lambda_v = lambda_vqp[:,:,0]	# shape - [batch_size, ans_len]
		lambda_q = lambda_vqp[:,:,1]
		lambda_p = lambda_vqp[:,:,2]

		P_y_temp1 = lambda_v[:,:,None] * P_v 	# shape - [batch_size, ans_len, ext_vocab_size]
		P_y_temp2 = lambda_q[:,:,None] * P_q
		P_y_temp3 = lambda_p[:,:,None] * P_p

		P_y = P_y_temp1 + P_y_temp2 + P_y_temp3		# shape - [batch_size, ans_len, ext_vocab_size]

		return P_y



class Model(nn.Module):
	def __init__(self, config, emb_matrix):
		super().__init__()
		self.config = config

		self.elmo = Elmo(self.config.elmo_options, self.config.elmo_weights, 1)

		self.reader_ranker_classifier = ReaderRankerClassifier(config, self.elmo)
		self.ans_decoder = AnswerDecoder(config, emb_matrix, self.elmo)


	def forward(self, e_q, e_p, ch_q, ch_p, qseq_len, q_words_idx, seq_len, p_words_idx, e_ans, ch_ans, aseq_len):
		# e : word embedding
		# c : character embedding
		# x_q - shape - [batch_size, q_len, emb_size]
		# x_p - shape - [batch_size, num_para, p_len, emb_size]
		# x_ans - shape - [batch_size, ans_len, emb_size]
		# qseq_len - shape - [batch_size]
		# seq_len - shape - [batch_size, num_para]
		# aseq_len - shape - [batch_size]
		# shape of q_words_idx - [batch_size, q_len]
		# shape of p_words_idx - [batch_size, num_para, p_len]


		beta_p, prob_ans, M_q, q_mask, M_p_k, p_mask = self.reader_ranker_classifier(e_q, e_p, ch_q, ch_p, qseq_len, seq_len)
		# shape of beta_p - [batch_size, num_para]
		# shape of prob_ans - [batch_size]
		# shape of M_q - [batch_size, q_len, hidden_size]
		# shape of q_mask- [batch_size, q_len]
		# shape of M_p_k - [batch_size, num_para, p_len, hidden_size]
		# shape of p_mask - [batch_size, num_para, p_len]


		P_y = self.ans_decoder(e_ans, ch_ans, aseq_len, M_q, q_mask, M_p_k, p_mask, beta_p, q_words_idx, p_words_idx)
		# shape of P_y - [batch_size, ans_len, ext_vocab_size]

		return beta_p, prob_ans, P_y
