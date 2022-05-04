import numpy as np
import argparse
import os
from pathlib import Path
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
import pickle
import pandas as pd
from video_speech_model import VideoSpeechModel
from emonet.emonet.models import EmoNetLSTM
from speech_emotion_classification_with_pytorch.models import ParallelCnnTransformerModel
from trigger_dataset import TriggerDataset
from sklearn.preprocessing import StandardScaler


def get_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--processed_data_dir",
        type=str, default="../processed_data",
        help="directory of processed data"
    )

    parser.add_argument(
        "--result_csv_path",
        type=str, default="../processed_data/result.csv",
        help="directory of processed data"
    )
    
    parser.add_argument(
        "--checkpoint_path",
        type=str, default="./outputs/multimodal_model/modelParams-2.pkl",
        help="Path to load checkpoint"
    )

    parser.add_argument(
        "--EmoNetLSTM_state_dict_path",
        type=str, default="./outputs/EmoNetLSTMfinetune/modelParams-1.pkl",
        help="checkpoint of pre-trained EmoNetLSTM model"
    )

    parser.add_argument(
        "--speech_state_dict_path",
        type=str, default="./speech_emotion_classification_with_pytorch/pretrained_models/cnn_transf_parallel_model.pt",
        help="checkpoint of pre-trained speech model"
    )

    parser.add_argument(
        "--scaler_dir",
        type=str, default="./speech_data_scaler.pkl",
        help="saved speech data scaler"
    )

    parser.add_argument(
        "--modality",
        type=str, default="multi",
        choices=['multi', 'video', 'audio'],
        help="Which modality to use"
    )

    parser.add_argument(
        "--device",
        type=str, default="gpu",
        help="Device type used for running the model"
    )

    parser.add_argument(
        "--batch_size",
        type=int, default=1,
        help="Batch size"
    )

    parser.add_argument(
        "--n_workers",
        type=int, default=1,
        help="Number of workers"
    )

    parser.add_argument(
        "--print_step_size",
        type=int, default=2,
        help="frequency of printing training summary"
    )

    parser.add_argument(
        "--checkpoints_save_path",
        type=str, default="./outputs/multimodal_model",
        help="Directory to save models"
    )


    args = parser.parse_args()
    return args


def inference_epoch(device, modality, model, data_loader, print_step_size, scaler, label_dict):
	model.eval()
	answer_df = pd.DataFrame(columns=['video_session', \
										'clip_idx', \
										'clip_name', \
										'clip_start_timestamp', \
										'clip_end_timestamp', \
										'student_pos',\
										'audio_path', \
										'frames_path', \
										'num_frames', \
										'label_idxs', \
										'labels'
										])
	for i, batch in enumerate(data_loader):
		if (i+1)%print_step_size==0:
			print("inference_step: {}/{}".format(i+1, len(data_loader)))
		img, speech, data_dict = batch
		if scaler:
			speech = batch_speech_transform(speech, scaler)
		img = img.to(device)
		speech = speech.to(device)
		if modality=='multi':
			output = model(img, speech)
		elif modality=='video':
			output = model(img)
		elif modality=='audio':
			output, output_softmax = model(speech)
		# print("output:\n", output, output.shape)
		label_idxs = torch.argmax(torch.softmax(output, dim=1), dim=1).detach().cpu().numpy()
		label_idxs = list(label_idxs)
		labels = [label_dict[idx] for idx in label_idxs]
		data_dict['label_idxs'] = label_idxs
		data_dict['labels'] = labels
		answer_df = answer_df.append(pd.DataFrame(data_dict), ignore_index = True)
		del img, speech, data_dict, output
		torch.cuda.empty_cache()

	return answer_df


def batch_speech_transform(batch_speech, scaler):
	b,c,h,w = batch_speech.shape
	batch_speech = np.reshape(batch_speech, newshape=(b,-1))
	batch_speech = scaler.transform(batch_speech)
	batch_speech = np.reshape(batch_speech, newshape=(b,c,h,w))
	batch_speech = torch.FloatTensor(batch_speech)
	return batch_speech


def main():
	args = get_parser()

	EMOTIONS = {1:'neutral', 2:'calm', 3:'happy', 4:'sad', 5:'angry', 6:'fear', 7:'disgust', 0:'surprise'}
	Emonet_emotions = {0:'neutral', 1:'happy', 2:'sad', 3:'surprise', 4:'fear', 5:'disgust', 6:'angry', 7:'calm'}

	n_expression = 8
	device = 'cuda' if torch.cuda.is_available() and args.device=='gpu' else 'cpu'
	print("device: ", device)
	
	dataset = TriggerDataset(args.processed_data_dir)
	dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.n_workers)

	scaler = None
	scaler_dir = args.scaler_dir
	if os.path.isfile(scaler_dir):
		with open(scaler_dir, 'rb') as f:
			scaler = pickle.load(f)
		print("Load speech scaler from {}.".format(scaler_dir))

	prediction_labels = EMOTIONS
	if args.modality == 'multi':
		model = VideoSpeechModel(n_expression=n_expression, 
							lstm_num_layer=2, 
							hidden_size=64, 
							output_size=n_expression, 
							dropout_rate=0.2, 
							EmoNetLSTM_state_dict_path=None,
							speech_num_emotions=len(EMOTIONS),
							speech_state_dict_path=None)
	elif args.modality == 'video':
		prediction_labels = Emonet_emotions
		state_dict_path = Path(__file__).parent.joinpath('emonet/pretrained', f'emonet_{n_expression}.pth')
		model = EmoNetLSTM(n_expression=n_expression, 
						lstm_num_layer=2, 
						hidden_size=64, 
						output_size=n_expression, 
						dropout_rate=0.2, 
						state_dict_path=state_dict_path)
	elif args.modality == 'audio':
		model = ParallelCnnTransformerModel(len(EMOTIONS))

	model = model.to(device)

	assert os.path.isfile(args.checkpoint_path)
	if device=='cpu':
		checkpoint_dict = torch.load(args.checkpoint_path, map_location=torch.device('cpu'))
	else:
		checkpoint_dict = torch.load(args.checkpoint_path)
	if args.modality == 'audio':
		model.load_state_dict(checkpoint_dict)
	else:
		model.load_state_dict(checkpoint_dict['state_dict'])
	print("Resuming model from: ", args.checkpoint_path)
	
	result = inference_epoch(device, args.modality, model, dataloader, args.print_step_size, scaler, prediction_labels)
	result.to_csv(args.result_csv_path, index=False)
	print("Inference Finished")


if __name__=='__main__':
	main()


