import numpy as np
from pathlib import Path
import argparse
import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data.sampler import WeightedRandomSampler
from torchvision import transforms 
import torch.nn.functional as F
import pickle
from emonet.emonet.models import EmoNet, EmoNetLSTM
from emonet.emonet.data import AffectNet
from emonet.emonet.data_augmentation import DataAugmentor
from emonet.emonet.metrics import CCC, PCC, RMSE, SAGR, ACC
from emonet.emonet.evaluation import evaluate, evaluate_flip
from ravdess_dataset import RavdessDataset, train_val_test_split, EMOTIONS, map_to_emonet_label
from video_speech_model import VideoSpeechModel
from sklearn.preprocessing import StandardScaler


def get_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--video_dir",
        type=str, default="./ravdess_data/videos",
        help="video folder directory"
    )

    parser.add_argument(
        "--speech_dir",
        type=str, default="./ravdess_data/audio_speech_actors_01-24/",
        help="speech folder directory"
    )

    parser.add_argument(
        "--landmark_dir",
        type=str, default="./ravdess_data/facial_landmarks",
        help="facial landmarks folder directory"
    )

    parser.add_argument(
        "--load_checkpoint",
        action="store_true",
        help="whether to load checkpoint from checkpoint dir"
    )
    parser.set_defaults(load_checkpoint=False)

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

    # RUNNING MODES:
    parser.add_argument(
        "--train",
        action="store_true",
        help="train the model using the training set"
    )
    parser.set_defaults(train=False)

    parser.add_argument(
        "--val",
        action="store_true",
        help="test the model using the validation set"
    )
    parser.set_defaults(val=False)

    parser.add_argument(
        "--device",
        type=str, default="gpu",
        help="Device type used for running the model"
    )

    # TRAINING PARAMETERS:
    parser.add_argument(
        "--num_train_epoches",
        type=int, default=30,
        help="Number of epoches to train"
    )

    parser.add_argument(
        "--batch_size",
        type=int, default=1,
        help="Batch size"
    )

    
    parser.add_argument(
        "--learning_rate",
        type=float, default=1e-4,
        help="Original learning rate"
    )

    parser.add_argument(
        "--weight_decay",
        type=float, default=5e-5,
        help="Weight decay coefficient for Adam"
    )

    parser.add_argument(
        "--train_print_step_size",
        type=int, default=20,
        help="frequency of printing training summary"
    )

    parser.add_argument(
        "--checkpoints_save_path",
        type=str, default="./outputs/multimodal_model",
        help="Directory to save models"
    )


    args = parser.parse_args()
    return args

def train_epoch(device, model, train_loader, optimizer, loss_cls, train_print_step_size, scaler):
	model.train()
	training_loss = 0.0
	training_correct = 0
	training_total = 0
	batch_loss = 0.0
	batch_correct = 0
	batch_total = 0
	for i, batch in enumerate(train_loader):
		optimizer.zero_grad()
		pad_img, speech, emonet_label, speech_label = batch
		speech = batch_speech_transform(speech, scaler)
		pad_img = pad_img.to(device)
		speech = speech.to(device)
		emonet_label = emonet_label.to(device)
		speech_label = speech_label.to(device)
		output = model(pad_img, speech)
		loss = loss_cls(output, speech_label)
		#print("train loss=", loss)
		loss.backward()
		optimizer.step()

		training_loss += loss.item()
		batch_loss += loss.item()
		_, predicted = torch.max(output.data, 1)
		batch_total += speech_label.size(0)
		training_total += speech_label.size(0)
		batch_correct += (predicted == speech_label).sum().item()
		training_correct += (predicted == speech_label).sum().item()
		if (i+1) % train_print_step_size == 0 or i+1 == len(train_loader):
			acc = float(batch_correct)/batch_total
			print('train_step: {}/{}, Loss: {}, Acc: {}'.format(i+1, len(train_loader), batch_loss/batch_total, batch_correct/batch_total))
			# print('train_step: {}/{}, Loss: {}'.format(i+1, len(train_loader), batch_loss/batch_total))

			batch_loss = 0.0
			batch_total = 0
			batch_correct = 0
		del pad_img, speech, emonet_label, speech_label, output, loss
		torch.cuda.empty_cache()

	return training_loss/training_total, training_correct/training_total


def eval_epoch(device, model, val_loader, scheduler, loss_cls, train_print_step_size, scaler):
	model.eval()
	correct = 0
	total = 0
	val_loss = 0.0
	for i, batch in enumerate(val_loader):
		if (i+1)%train_print_step_size==0:
			print("eval_step: {}/{}".format(i+1, len(val_loader)))
		pad_img, speech, emonet_label, speech_label = batch
		speech = batch_speech_transform(speech, scaler)
		pad_img = pad_img.to(device)
		speech = speech.to(device)
		emonet_label = emonet_label.to(device)
		speech_label = speech_label.to(device)
		output = model(pad_img, speech)
		loss = loss_cls(output, speech_label)

		_, predicted = torch.max(output.data, 1)
		total += speech_label.size(0)
		correct += (predicted == speech_label).sum().item()
		val_loss += loss.item()
		del pad_img, speech, emonet_label, speech_label, output, loss
		torch.cuda.empty_cache()

	val_loss /= total
	if scheduler:
		scheduler.step(val_loss)

	return val_loss, correct/total


def data_scaler(train_dataset):
	print("Data scaling ... ")
	X_train = []
	for i, data in enumerate(train_dataset):
		img, speech, emonet_label, speech_label = data
		X_train.append(speech)
		print("scaling {}/{}".format(i, len(train_dataset)))
	X_train = np.array(torch.stack(X_train, dim=0))
	# X_train = np.expand_dims(X_train,1)
	

	scaler = StandardScaler()

	b,c,h,w = X_train.shape
	X_train = np.reshape(X_train, newshape=(b,-1))
	X_train = scaler.fit_transform(X_train)
	print("Data scaling done.")
	return scaler

def batch_speech_transform(batch_speech, scaler):
	b,c,h,w = batch_speech.shape
	batch_speech = np.reshape(batch_speech, newshape=(b,-1))
	batch_speech = scaler.transform(batch_speech)
	batch_speech = np.reshape(batch_speech, newshape=(b,c,h,w))
	batch_speech = torch.FloatTensor(batch_speech)
	return batch_speech


def main():
	args = get_parser()

	video_dir = args.video_dir
	speech_dir = args.speech_dir
	landmark_dir = args.landmark_dir

	# checkpoints_save_path = './outputs/multimodal_model'
	# load_checkpoint = True
	# checkpoint_path = './outputs/multimodal_model/modelParams-3.pkl'

	# EmoNetLSTM_state_dict_path = './outputs/EmoNetLSTMfinetune/modelParams-1.pkl'
	# speech_state_dict_path = './speech_emotion_classification_with_pytorch/pretrained_models/cnn_transf_parallel_model.pt'

	image_size = 256
	n_expression = 8
	device = 'cuda' if torch.cuda.is_available() or args.device=='gpu' else 'cpu'

	n_workers = 8
	transform_image = transforms.Compose([transforms.ToTensor()])
	transform_image_shape_no_flip = DataAugmentor(image_size, image_size)
	# flipping_indices = [16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0, 26, 25, 24, 23, 22,21, 20, 19, 18, 17, 27, 28, 29, 30, 35, 34, 33, 32, 31, 45,44, 43, 42, 47, 46, 39, 38, 37, 36, 41, 40, 54, 53, 52, 51,50, 49, 48, 59, 58,57, 56, 55, 64, 63,62, 61, 60, 67, 66,65]
	# transform_image_shape_flip = DataAugmentor(image_size, image_size, mirror=True, shape_mirror_indx=flipping_indices, flipping_probability=1.0)

	train_ids, val_ids, test_ids = train_val_test_split(video_dir)
	train_dataset = RavdessDataset(video_dir, speech_dir, landmark_dir, train_ids, 
						transform_image_shape=transform_image_shape_no_flip, transform_image=transform_image)
	val_dataset = RavdessDataset(video_dir, speech_dir, landmark_dir, val_ids, 
						transform_image_shape=transform_image_shape_no_flip, transform_image=transform_image)
	test_dataset = RavdessDataset(video_dir, speech_dir, landmark_dir, test_ids, 
						transform_image_shape=transform_image_shape_no_flip, transform_image=transform_image)
	train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=n_workers)
	val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=n_workers)
	test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=n_workers)

	scaler_dir = args.scaler_dir
	if os.path.isfile(scaler_dir):
		with open(scaler_dir, 'rb') as f:
			scaler = pickle.load(f)
		print("Load speech scaler from {}.".format(scaler_dir))
	else:
		scaler = data_scaler(train_dataset)
		with open(scaler_dir, 'wb') as f:
			pickle.dump(scaler, f)

	# Loading the model 
	state_dict_path = Path(__file__).parent.joinpath('emonet/pretrained', f'emonet_{n_expression}.pth')

	# print(f'Loading the model from {state_dict_path}.')
	# state_dict = torch.load(str(state_dict_path), map_location='cpu')
	# state_dict = {k.replace('module.',''):v for k,v in state_dict.items()}
	# net = EmoNet(n_expression=n_expression).to(device)
	# net.load_state_dict(state_dict, strict=False)
	# net.eval()
	

	model = VideoSpeechModel(n_expression=n_expression, 
						lstm_num_layer=2, 
						hidden_size=64, 
						output_size=n_expression, 
						dropout_rate=0.2, 
						EmoNetLSTM_state_dict_path=args.EmoNetLSTM_state_dict_path,
						speech_num_emotions=len(EMOTIONS),
						speech_state_dict_path=args.speech_state_dict_path)

	model = model.to(device)

	loss_cls = nn.CrossEntropyLoss()
	optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
	scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=1)

	if args.load_checkpoint:
		assert os.path.isfile(args.checkpoint_path)
		print("Resuming model from: ", args.checkpoint_path)
		checkpoint_dict = torch.load(args.checkpoint_path)
		model.load_state_dict(checkpoint_dict['state_dict'])
		optimizer.load_state_dict(checkpoint_dict['optimizer'])
		start_epoch = checkpoint_dict['epoch']
		

	else:
		start_epoch = 0
		if os.path.isdir(args.checkpoints_save_path) == False:
			os.mkdir(args.checkpoints_save_path)

		model_checkpoint_path = os.path.join(args.checkpoints_save_path, 'modelParams-init.pkl')
		checkpoint_dict = {'state_dict': model.state_dict(),
							'optimizer': optimizer.state_dict(), 
							'epoch': start_epoch}
		torch.save(checkpoint_dict, model_checkpoint_path)
		print("Initial model saved to "+model_checkpoint_path)
		# print("Start Training ... ")

	if args.val:
		avg_eval_loss, avg_eval_acc = eval_epoch(device, model, val_dataloader, None, loss_cls, args.train_print_step_size, scaler)
		print('Eval - Ave Loss: {}, Ave Accuracy: {}'.format(avg_eval_loss, avg_eval_acc))



	if args.train:
		print("Start Training from epoch: ", start_epoch)
		best_eval_loss = None
		best_eval_acc = 0
		for epoch in range(start_epoch, args.num_train_epoches):
			avg_training_loss, avg_training_acc = train_epoch(device, model, train_dataloader, optimizer, loss_cls, args.train_print_step_size, scaler)
			print('Train - Epoch: [{}/{}] Done, Ave Loss: {}, Ave Accuracy: {}'.format(epoch+1, args.num_train_epoches, avg_training_loss, avg_training_acc))


			avg_eval_loss, avg_eval_acc = eval_epoch(device, model, val_dataloader, scheduler, loss_cls, args.train_print_step_size, scaler)
			print('Eval - Epoch: [{}/{}] Done, Ave Loss: {}, Ave Accuracy: {}'.format(epoch+1, args.num_train_epoches, avg_eval_loss, avg_eval_acc))

			if best_eval_loss==None:
				best_eval_loss = avg_eval_loss
			if avg_eval_loss<best_eval_loss or avg_eval_acc>best_eval_acc:
				if os.path.isdir(args.checkpoints_save_path) == False:
					os.mkdir(args.checkpoints_save_path)
				model_checkpoint_path = os.path.join(args.checkpoints_save_path, 'modelParams-'+str(epoch+1)+'.pkl')
				checkpoint_dict = {'state_dict': model.state_dict(),
									'optimizer': optimizer.state_dict(), 
									'epoch':epoch+1}
				torch.save(checkpoint_dict, model_checkpoint_path)
				print("Model saved to "+model_checkpoint_path)
				best_eval_loss = min(best_eval_loss, avg_eval_loss)
				best_eval_acc = max(best_eval_acc, avg_eval_acc)


if __name__=='__main__':
	main()


