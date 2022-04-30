from pathlib import Path 
import pickle
import numpy as np
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import pandas as pd
import torch
import math
from torch.utils.data import Dataset
from skimage import io
import os
import librosa
import cv2
from emonet.emonet.data_augmentation import DataAugmentor
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence,pack_padded_sequence,pack_sequence,pad_packed_sequence
from sklearn.preprocessing import StandardScaler


EMOTIONS = {1:'neutral', 2:'calm', 3:'happy', 4:'sad', 5:'angry', 6:'fear', 7:'disgust', 0:'surprise'}
	
# _expressions = {0: 'neutral', 1:'happy', 2:'sad', 3:'surprise', 4:'fear', 5:'disgust', 6:'anger', 7:'contempt', 8:'none'}
map_to_emonet_label = {'neutral':0, 'happy':1, 'sad':2, 'surprise':3, 'fear':4, 'disgust':5, 'angry':6, 'calm':7}
SAMPLE_RATE = 48000

def addAWGN(signal, num_bits=16, augmented_num=2, snr_low=15, snr_high=30): 
	signal_len = len(signal)
	# Generate White Gaussian noise
	noise = np.random.normal(size=(augmented_num, signal_len))
	# Normalize signal and noise
	norm_constant = 2.0**(num_bits-1)
	signal_norm = signal / norm_constant
	noise_norm = noise / norm_constant
	# Compute signal and noise power
	s_power = np.sum(signal_norm ** 2) / signal_len
	n_power = np.sum(noise_norm ** 2, axis=1) / signal_len
	# Random SNR: Uniform [15, 30] in dB
	target_snr = np.random.randint(snr_low, snr_high)
	# Compute K (covariance matrix) for each noise 
	K = np.sqrt((s_power / n_power) * 10 ** (- target_snr / 10))
	K = np.ones((signal_len, augmented_num)) * K  
	# Generate noisy signal
	return signal + K.T * noise

def getMELspectrogram(audio, sample_rate):
	mel_spec = librosa.feature.melspectrogram(y=audio,
											sr=sample_rate,
											n_fft=1024,
											win_length = 512,
											window='hamming',
											hop_length = 256,
											n_mels=128,
											fmax=sample_rate/2
											)
	mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
	return mel_spec_db



def train_val_test_split(speech_dir):
	ids = []
	train_ids, val_ids, test_ids = [], [], []
	data = pd.DataFrame(columns=['Data_ID', 'Emotion'])
	for dirname, _, filenames in os.walk(speech_dir):
		for filename in filenames:

			key = filename.split('.')[0]
			identifiers = key.split('-')
			data_id = '-'.join(identifiers[1:])
			emotion = (int(identifiers[2]))
			if emotion == 8: # promeni surprise sa 8 na 0
				emotion = 0
			data = data.append({"Data_ID": data_id,
							"Emotion": emotion
							},
							ignore_index = True
						)
			ids.append(data_id)
	# print(len(ids))

	for emotion in range(len(EMOTIONS)):
		emotion_ind = list(data.loc[data.Emotion==emotion,'Emotion'].index)
		emotion_ind = np.random.permutation(emotion_ind)
		m = len(emotion_ind)
		# print(emotion_ind)
		ind_train = emotion_ind[:int(0.8*m)]
		ind_val = emotion_ind[int(0.8*m):int(0.9*m)]
		ind_test = emotion_ind[int(0.9*m):]
		train_ids.extend([ids[i] for i in ind_train])
		val_ids.extend([ids[i] for i in ind_val])
		test_ids.extend([ids[i] for i in ind_test])

	print("train/val/test split: ", len(train_ids), len(val_ids), len(test_ids))
	# print(val_ids)
	return train_ids, val_ids, test_ids

class RavdessDataset(Dataset):
	

	def __init__(self, video_dir, speech_dir, landmark_dir, id_set, transform_image_shape=None, transform_image=None):
		
		data = pd.DataFrame(columns=['data_id', 'emonet_label', 'speech_label', 'video_path', 'speech_path', 'landmark_path'])
		for data_id in id_set:
			vocal_channel, emotion, emotion_intensity, statement, repetition, actor = data_id.split('-')
			video_path = os.path.join(video_dir, 'Actor_'+actor, '02-'+data_id+'.mp4')
			emotion = int(emotion)
			if emotion == 8: # promeni surprise sa 8 na 0
				emotion = 0
			emonet_label = map_to_emonet_label[EMOTIONS[emotion]]

			landmark_path = os.path.join(landmark_dir, '01-'+data_id+'.csv')
			speech_file_path = os.path.join(speech_dir, 'Actor_'+actor, '03-'+data_id+'.wav')

			data = data.append({"data_id": data_id,
						"emonet_label": emonet_label,
						"speech_label": emotion, 
						"video_path": video_path,
						"speech_path": speech_file_path,
						"landmark_path": landmark_path
						},
						ignore_index = True
					)

		self.data = data
		self.landmark_cols = ['x_'+str(i) for i in range(0, 68)] + ['y_'+str(i) for i in range(0, 68)]
		self.transform_image = transform_image
		self.transform_image_shape = transform_image_shape

	
	def __len__(self):
		return self.data.shape[0]

	def __getitem__(self, index):
		single_data = self.data.iloc[[index]].to_dict('r')[0]
		images = self.get_images(single_data['video_path'], single_data['landmark_path'])
		speech = self.get_speech(single_data['speech_path'])
		emonet_label = torch.tensor(single_data['emonet_label'])
		speech_label = torch.tensor(single_data['speech_label'])
		return images, speech, emonet_label, speech_label

	
	def get_images(self, video_path, landmark_path):
		# print("video_path: ", video_path)
		vidcap = cv2.VideoCapture(video_path)
		images = []
		success,image = vidcap.read()
		while success:
			images.append(image)
			success,image = vidcap.read()
		
		lm_df = pd.read_csv(landmark_path)
		# print("len(images), lm_df.shape[0]: ", len(images), lm_df.shape[0])
		if lm_df.shape[0]!=len(images):
			min_num = min(lm_df.shape[0], len(images))
			lm_df = lm_df[:min_num]
			images = images[:min_num]
		assert lm_df.shape[0]==len(images)

		ind = list(lm_df.loc[(lm_df['timestamp']>=0.5) & (lm_df['timestamp']<=3.5)].index)
		res_images = [images[i] for i in ind]
		lm_df = lm_df.loc[(lm_df['timestamp']>=0.5) & (lm_df['timestamp']<=3.5)]
		lm = np.array(lm_df[self.landmark_cols])
		res_lm = np.stack([lm[:, :68], lm[:, 68:]], axis=-1)
		# print("res_images, res_lm: ", len(res_images), res_lm.shape)
		assert len(res_images)==res_lm.shape[0]

		aug_images = []
		for i in range(len(res_images)):
			img = res_images[i]
			landmarks = res_lm[i, :]
			# print("img, landmarks: ", img.shape, landmarks.shape)
			if self.transform_image_shape is not None:
				bounding_box = [landmarks.min(axis=0)[0], landmarks.min(axis=0)[1],
								landmarks.max(axis=0)[0], landmarks.max(axis=0)[1]]
				#image, landmarks = self.transform_image_shape(image, shape=landmarks)
				img, landmarks = self.transform_image_shape(img, bb=bounding_box)
				# Fix for PyTorch currently not supporting negative stric
				img = np.ascontiguousarray(img)

			# print("transform_image_shape: ", type(img), img.shape)
			if self.transform_image is not None:
				img = self.transform_image(img)
			# print("transform_image: ", type(img), img.shape)
			aug_images.append(img)
		aug_images = np.stack(aug_images, axis=0)
		B, C, H, W = aug_images.shape
		out_images = np.zeros((30*3, C, H, W))
		if B < 30*3:
			out_images[:B] = aug_images
		else:
			out_images = aug_images[:30*3]
		out_images = torch.FloatTensor(out_images)
		return out_images


	def get_speech(self, speech_path):
		audio, sample_rate = librosa.load(speech_path, duration=3, offset=0.5, sr=SAMPLE_RATE)
		# print("audio: ", audio.shape)
		signal = np.zeros((int(SAMPLE_RATE*3,)))
		signal[:len(audio)] = audio
		# print("signal: ", signal.shape)
		augmented_signals = addAWGN(signal)
		# print("augmented_signals: ", augmented_signals.shape)
		mel_spectrogram = getMELspectrogram(signal, SAMPLE_RATE)
		# print("mel_spectrogram: ", mel_spectrogram.shape)
		mel_spectrogram = torch.FloatTensor(np.expand_dims(mel_spectrogram, 0))
		return mel_spectrogram

	# def collate_fn(batch):
	# 	batch.sort(key=lambda x: len(x[0]), reverse=True)
	# 	img, speech, emonet_label, speech_label = zip(*batch)
		
	# 	img = list(img)
	# 	speech = list(speech)
	# 	emonet_label = list(emonet_label)
	# 	speech_label = list(speech_label)
	# 	img_lens = torch.LongTensor([len(i) for i in img])
	# 	emonet_label = torch.tensor(emonet_label)
	# 	speech_label = torch.tensor(speech_label)
	# 	speech = torch.FloatTensor(np.stack(speech, axis=0))
	# 	pad_img = pad_sequence(img, batch_first=True, padding_value=0)
	# 	return pad_img, speech, emonet_label, speech_label, img_lens


if __name__ == "__main__":
	video_dir = './ravdess_data/videos'
	speech_dir = './ravdess_data/audio_speech_actors_01-24/'
	landmark_dir = './ravdess_data/facial_landmarks'

	image_size = 256
	batch_size = 3
	n_workers = 1
	transform_image = transforms.Compose([transforms.ToTensor()])
	transform_image_shape_no_flip = DataAugmentor(image_size, image_size)
	flipping_indices = [16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0, 26, 25, 24, 23, 22,21, 20, 19, 18, 17, 27, 28, 29, 30, 35, 34, 33, 32, 31, 45,44, 43, 42, 47, 46, 39, 38, 37, 36, 41, 40, 54, 53, 52, 51,50, 49, 48, 59, 58,57, 56, 55, 64, 63,62, 61, 60, 67, 66,65]
	transform_image_shape_flip = DataAugmentor(image_size, image_size, mirror=True, shape_mirror_indx=flipping_indices, flipping_probability=1.0)

	train_ids, val_ids, test_ids = train_val_test_split(video_dir)
	dataset = RavdessDataset(video_dir, speech_dir, landmark_dir, test_ids, 
						transform_image_shape=transform_image_shape_no_flip, transform_image=transform_image)
	# test_dataloader_no_flip = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=n_workers)
	# for batch in test_dataloader_no_flip:
	# 	img, speech, emonet_label, speech_label = batch
	# 	print("type: ", type(img), type(speech), type(emonet_label), type(speech_label))
	# 	print(img.shape, speech.shape, emonet_label.shape, speech_label.shape)
	# 	print(emonet_label)
	# 	print(speech_label)
	speech_data = []
	for data in dataset:
		img, speech, emonet_label, speech_label = data
		speech_data.append(speech) #[128, 563]
		
	scaler = StandardScaler()
	speech_data = np.array(torch.stack(speech_data, dim=0))
	speech_data = np.expand_dims(speech_data,1)
	print(speech_data.shape)

	b,c,h,w = speech_data.shape

	speech_data = np.reshape(speech_data, newshape=(b,-1))
	print(speech_data[0])
	speech_data = scaler.fit_transform(speech_data)
	speech_data = np.reshape(speech_data, newshape=(b,c,h,w))
	print(speech_data[0])

	single_data = np.reshape(np.array(speech), newshape=(1,-1))

	single_tranform = scaler.transform(single_data)
	print(single_tranform.shape, single_tranform)
	single_tranform = np.reshape(single_tranform, newshape=(1,1,h,w))
	print("single_tranform: ", single_tranform)
	print("scaler single transform: ", scaler.transform(speech))






