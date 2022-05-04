from emonet.emonet.models import EmoNet, EmoNetLSTM
from speech_emotion_classification_with_pytorch.models import ParallelCnnTransformerModel

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np


class VideoSpeechModel(nn.Module):
	def __init__(self, n_expression, lstm_num_layer, hidden_size, output_size,
					dropout_rate=0.2, EmoNetLSTM_state_dict_path=None,
					speech_num_emotions=8, speech_state_dict_path=None):
		super(VideoSpeechModel, self).__init__()

		self.emonet_lstm = EmoNetLSTM(n_expression=n_expression, 
						lstm_num_layer=lstm_num_layer, 
						hidden_size=hidden_size, 
						output_size=output_size, 
						dropout_rate=dropout_rate, 
						state_dict_path=None)

		if EmoNetLSTM_state_dict_path:
			checkpoint_dict = torch.load(EmoNetLSTM_state_dict_path, map_location=torch.device('cpu'))
			self.emonet_lstm.load_state_dict(checkpoint_dict['state_dict'])
			print(f'Load the pretrained emonet from {EmoNetLSTM_state_dict_path}.')

		self.speech_net = ParallelCnnTransformerModel(speech_num_emotions)
		if speech_state_dict_path:
			speech_checkpoint = torch.load(speech_state_dict_path, map_location=torch.device('cpu'))
			self.speech_net.load_state_dict(speech_checkpoint)
			print(f'Load the pretrained speech_net from {speech_state_dict_path}.')

		self.fc = nn.Linear(output_size*2, output_size)

	def forward(self, img, speech):
		emonet_output = self.emonet_lstm(img)
		speech_output_logits, speech_output_softmax = self.speech_net(speech)
		# print("emonet_output: ", emonet_output.shape)
		# print("speech_output_logits: ", speech_output_logits.shape)
		out = torch.cat([emonet_output, speech_output_logits], dim=1)
		out = self.fc(out)
		# print("out: ", out.shape)
		return out



