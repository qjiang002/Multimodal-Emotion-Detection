from pathlib import Path 
import pickle
import numpy as np
import os
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import pandas as pd
import torch
import math
import librosa
from torch.utils.data import Dataset
from torchvision import transforms
from torch.utils.data import DataLoader
from skimage import io


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

class TriggerDataset(Dataset):
    def __init__(self,
                 processed_data_dir,
                 clip_duration = 3,
                 video_fps = 30,
                 audio_sample_rate = 48000):
        self.processed_data_dir = processed_data_dir
        self.clip_duration = clip_duration
        self.video_fps = video_fps
        self.audio_sample_rate = audio_sample_rate
        self.transform_image = transforms.Compose([transforms.ToTensor()])
        self.data = self.generate_clip_df(processed_data_dir)
        
        
    def __len__(self):
        return self.data.shape[0]

    def generate_clip_df(self, processed_data_dir):
        data = pd.DataFrame(columns=['video_session', \
                                     'clip_idx', \
                                     'clip_name', \
                                     'clip_start_timestamp', \
                                     'clip_end_timestamp', \
                                     'student_pos',\
                                     'audio_path', \
                                     'frames_path', \
                                     'num_frames'
                                    ])
        for video_session in os.listdir(processed_data_dir):
            video_session_dir = os.path.join(processed_data_dir, video_session)
            if not os.path.isdir(video_session_dir):
                continue
            for clip_name in os.listdir(video_session_dir):
                clip_dir = os.path.join(video_session_dir, clip_name)
                clip_idx = int(clip_name.split('_')[-1])
                clip_start_timestamp = clip_idx * self.clip_duration
                clip_end_timestamp = (clip_idx+1) * self.clip_duration
                audio_path = os.path.join(clip_dir, 'audio_clip.wav')
                for student_frame in os.listdir(clip_dir):
                    if not student_frame.endswith('frames'):
                        continue
                    student_pos = student_frame.split('_')[0]
                    frames_path = os.path.join(clip_dir, student_frame)
                    num_frames = len(os.listdir(frames_path))
                    data = data.append({"video_session": video_session,
                                         'clip_idx': clip_idx,
                                         'clip_name': clip_name,
                                         'clip_start_timestamp': clip_start_timestamp,
                                         'clip_end_timestamp': clip_end_timestamp,
                                         'student_pos': student_pos,
                                         'audio_path': audio_path,
                                         'frames_path': frames_path,
                                         'num_frames': num_frames
                                        },
                                        ignore_index = True
                                    )
        return data
    
    def __getitem__(self, index):
        single_data = self.data.iloc[[index]].to_dict('r')[0]
        images = self.get_images(single_data['frames_path'], single_data['num_frames'])
        speech = self.get_audio(single_data['audio_path'])
        return images, speech, single_data
    
    def get_images(self, frames_path, num_frames):
        images = []
        for frame_idx in range(num_frames):
            img = io.imread(os.path.join(frames_path, 'frame_{}.jpg'.format(frame_idx)))
            img = np.ascontiguousarray(img)
            img = self.transform_image(img)
            images.append(img)
        images = np.stack(images, axis=0)
        B, C, H, W = images.shape
        out_B = self.video_fps*self.clip_duration
        out_images = np.zeros((out_B, C, H, W))
        if B < out_B:
            out_images[:B] = images
        else:
            out_images = images[:out_B]
        out_images = torch.FloatTensor(out_images)
        return out_images
    
    def get_audio(self, audio_path):
        audio, sample_rate = librosa.load(audio_path, duration=self.clip_duration, sr=self.audio_sample_rate)
        signal = np.zeros((int(self.audio_sample_rate*self.clip_duration,)))
        signal[:len(audio)] = audio
        augmented_signals = addAWGN(signal)
        mel_spectrogram = getMELspectrogram(signal, self.audio_sample_rate)
        mel_spectrogram = torch.FloatTensor(np.expand_dims(mel_spectrogram, 0))
        return mel_spectrogram


if __name__ == "__main__":
    processed_data_dir = '../processed_data'
    data_set = TriggerDataset(processed_data_dir)
    for data in data_set:
        img, speech, data_dict = data
        print(img.shape, speech.shape)
        print(data_dict)
        break
    # torch.Size([90, 3, 256, 256]) torch.Size([1, 128, 563])
    # {'video_session': 'demosharp', 'clip_idx': 17, 'clip_name': 'clip_17', 'clip_start_timestamp': 51, 'clip_end_timestamp': 54, 'student_pos': 'right', 'audio_path': '../processed_data/demosharp/clip_17/audio_clip.wav', 'frames_path': '../processed_data/demosharp/clip_17/right_frames', 'num_frames': 90}
    data_loader = DataLoader(data_set, batch_size=2, shuffle=False, num_workers=1)
    for i,batch in enumerate(data_loader):
        img, speech, data_dict = batch
        print(img.shape, speech.shape, data_dict)
        if i>=5:
            break
    # torch.Size([2, 90, 3, 256, 256]) torch.Size([2, 1, 128, 563]) 
    # {'video_session': ['demosharp', 'demosharp'], 'clip_idx': tensor([17, 17]), 'clip_name': ['clip_17', 'clip_17'], 'clip_start_timestamp': tensor([51, 51]), 'clip_end_timestamp': tensor([54, 54]), 'student_pos': ['right', 'mid'], 'audio_path': ['../processed_data/demosharp/clip_17/audio_clip.wav', '../processed_data/demosharp/clip_17/audio_clip.wav'], 'frames_path': ['../processed_data/demosharp/clip_17/right_frames', '../processed_data/demosharp/clip_17/mid_frames'], 'num_frames': tensor([90, 90])}



