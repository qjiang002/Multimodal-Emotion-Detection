from .emonet import EmoNet
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence

class EmoNetLSTM(nn.Module):
	def __init__(self, n_expression, lstm_num_layer, hidden_size, output_size, dropout_rate=0.2, state_dict_path=None):
		super(EmoNetLSTM, self).__init__()

		self.n_expression = n_expression
		self.hidden_size = hidden_size
		self.output_size = output_size
		self.dropout_rate = dropout_rate
		self.lstm_num_layer = lstm_num_layer
		
		self.net = EmoNet(n_expression=n_expression)
		if state_dict_path:
			state_dict = torch.load(str(state_dict_path), map_location='cpu')
			state_dict = {k.replace('module.',''):v for k,v in state_dict.items()}
			self.net.load_state_dict(state_dict, strict=False)
			print(f'Load the pretrained emonet from {state_dict_path}.')

		self.rnns = nn.LSTM(input_size=n_expression,
							hidden_size=hidden_size,
							num_layers=lstm_num_layer,
							bias=True,
							batch_first=True,
							dropout=dropout_rate,
							bidirectional=True)
		self.fc = torch.nn.Sequential(
							nn.Linear(90*hidden_size*2, hidden_size),
							nn.Linear(hidden_size, output_size))

	def forward(self, img): 
		B, L, C, W, H = img.shape
		img = img.view(B*L, C, W, H)
		net_out = self.net(img)
		expression = net_out['expression'] # (B*max_l, 8)
		# print("expression: ", expression.shape)
		expression = expression.view(B, L, -1)
		# print("expression: ", expression.shape)
		rnn_out, hidden = self.rnns(expression)
		# print("rnn_out: ", rnn_out.shape)
		rnn_out = rnn_out.view(B, -1)
		out_prob = self.fc(rnn_out)
		# print("out_prob: ", out_prob.shape, out_prob)
		return out_prob

