from speech_data_processing import speechDataProcessing, EMOTIONS, DATA_PATH
from models import ParallelCnnTransformerModel

from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn

import numpy as np
import os
import pickle


def loss_fnc(predictions, targets):
	return nn.CrossEntropyLoss()(input=predictions,target=targets)


def make_train_step(model, loss_fnc, optimizer):
    def train_step(X,Y):
        # set model to train mode
        model.train()
        # forward pass
        output_logits, output_softmax = model(X)
        predictions = torch.argmax(output_softmax,dim=1)
        accuracy = torch.sum(Y==predictions)/float(len(Y))
        # compute loss
        loss = loss_fnc(output_logits, Y)
        # compute gradients
        loss.backward()
        # update parameters and zero gradients
        optimizer.step()
        optimizer.zero_grad()
        return loss.item(), accuracy*100
    return train_step

def make_validate_fnc(model,loss_fnc):
    def validate(X,Y):
        with torch.no_grad():
            model.eval()
            output_logits, output_softmax = model(X)
            predictions = torch.argmax(output_softmax,dim=1)
            accuracy = torch.sum(Y==predictions)/float(len(Y))
            loss = loss_fnc(output_logits,Y)
        return loss.item(), accuracy*100, predictions
    return validate


def data_scaler(X_train, X_val, X_test):
	print("Data scaling ... ")
	X_train = np.expand_dims(X_train,1)
	X_val = np.expand_dims(X_val,1)
	X_test = np.expand_dims(X_test,1)

	scaler = StandardScaler()

	b,c,h,w = X_train.shape
	X_train = np.reshape(X_train, newshape=(b,-1))
	X_train = scaler.fit_transform(X_train)
	X_train = np.reshape(X_train, newshape=(b,c,h,w))

	b,c,h,w = X_test.shape
	X_test = np.reshape(X_test, newshape=(b,-1))
	X_test = scaler.transform(X_test)
	X_test = np.reshape(X_test, newshape=(b,c,h,w))

	b,c,h,w = X_val.shape
	X_val = np.reshape(X_val, newshape=(b,-1))
	X_val = scaler.transform(X_val)
	X_val = np.reshape(X_val, newshape=(b,c,h,w))

	return X_train, X_val, X_test

def train(X_train, X_val, Y_train, Y_val):
	EPOCHS=1500
	DATASET_SIZE = X_train.shape[0]
	BATCH_SIZE = 32
	device = 'cuda' if torch.cuda.is_available() else 'cpu'
	print('Selected device is {}'.format(device))
	model = ParallelCnnTransformerModel(num_emotions=len(EMOTIONS)).to(device)
	print('Number of trainable params: ',sum(p.numel() for p in model.parameters()) )
	OPTIMIZER = torch.optim.SGD(model.parameters(),lr=0.01, weight_decay=1e-3, momentum=0.8)

	train_step = make_train_step(model, loss_fnc, optimizer=OPTIMIZER)
	validate = make_validate_fnc(model,loss_fnc)
	losses=[]
	val_losses = []
	for epoch in range(EPOCHS):
		# schuffle data
		ind = np.random.permutation(DATASET_SIZE)
		X_train = X_train[ind,:,:,:]
		Y_train = Y_train[ind]
		epoch_acc = 0
		epoch_loss = 0
		iters = int(DATASET_SIZE / BATCH_SIZE)
		for i in range(iters):
			batch_start = i * BATCH_SIZE
			batch_end = min(batch_start + BATCH_SIZE, DATASET_SIZE)
			actual_batch_size = batch_end-batch_start
			X = X_train[batch_start:batch_end,:,:,:]
			Y = Y_train[batch_start:batch_end]
			X_tensor = torch.tensor(X,device=device).float()
			Y_tensor = torch.tensor(Y, dtype=torch.long,device=device)
			loss, acc = train_step(X_tensor,Y_tensor)
			epoch_acc += acc*actual_batch_size/DATASET_SIZE
			epoch_loss += loss*actual_batch_size/DATASET_SIZE
			print(f"\r Epoch {epoch}: iteration {i}/{iters}",end='')
		X_val_tensor = torch.tensor(X_val,device=device).float()
		Y_val_tensor = torch.tensor(Y_val,dtype=torch.long,device=device)
		val_loss, val_acc, predictions = validate(X_val_tensor,Y_val_tensor)
		losses.append(epoch_loss)
		val_losses.append(val_loss)
		print('')
		print(f"Epoch {epoch} --> loss:{epoch_loss:.4f}, acc:{epoch_acc:.2f}%, val_loss:{val_loss:.4f}, val_acc:{val_acc:.2f}%")
	SAVE_PATH = os.path.join(os.getcwd(),'pretrained_models')
	os.makedirs('pretrained_models',exist_ok=True)
	torch.save(model.state_dict(),os.path.join(SAVE_PATH,'cnn_transf_parallel_model.pt'))
	print('Model is saved to {}'.format(os.path.join(SAVE_PATH,'cnn_transf_parallel_model.pt')))


def test(model, X_test, Y_test):
	device = 'cuda' if torch.cuda.is_available() else 'cpu'
	print("device = ", device)
	model.to(device)
	validate = make_validate_fnc(model,loss_fnc)
	X_test_tensor = torch.tensor(X_test,device=device).float()
	Y_test_tensor = torch.tensor(Y_test,dtype=torch.long,device=device)
	test_loss, test_acc, predictions = validate(X_test_tensor,Y_test_tensor)
	print(f'Test loss is {test_loss:.3f}')
	print(f'Test accuracy is {test_acc:.2f}%')
	

def load_model():
	LOAD_PATH = os.path.join(os.getcwd(),'pretrained_models')
	model = ParallelCnnTransformerModel(len(EMOTIONS))
	model.load_state_dict(torch.load(os.path.join(LOAD_PATH,'cnn_transf_parallel_model.pt')))
	print('Model is loaded from {}'.format(os.path.join(LOAD_PATH,'cnn_transf_parallel_model.pt')))
	return model

def load_dataset():
	print("Load datasets ... ")

	try:
		processed_data_dir = './processed_data'
		with open(os.path.join(processed_data_dir, 'X_train.pkl'), 'rb') as f:
			X_train = pickle.load(f)
		with open(os.path.join(processed_data_dir, 'X_val.pkl'), 'rb') as f:
			X_val = pickle.load(f)
		with open(os.path.join(processed_data_dir, 'X_test.pkl'), 'rb') as f:
			X_test = pickle.load(f)
		with open(os.path.join(processed_data_dir, 'Y_train.pkl'), 'rb') as f:
			Y_train = pickle.load(f)
		with open(os.path.join(processed_data_dir, 'Y_val.pkl'), 'rb') as f:
			Y_val = pickle.load(f)
		with open(os.path.join(processed_data_dir, 'Y_test.pkl'), 'rb') as f:
			Y_test = pickle.load(f)
	except:
		X_train, X_val, X_test, Y_train, Y_val, Y_test = speechDataProcessing()
	return X_train, X_val, X_test, Y_train, Y_val, Y_test


def speech_train(train_flag=False, test_flag=True):
	X_train, X_val, X_test, Y_train, Y_val, Y_test = load_dataset()
	print(X_train.shape, Y_train.shape)
	X_train, X_val, X_test = data_scaler(X_train, X_val, X_test)
	print("scaled: ", X_train.shape)
	if train_flag:
		train(X_train, X_val, Y_train, Y_val)
	elif test_flag:
		model = load_model()
		test(model, X_test, Y_test)


if __name__ == "__main__":
	speech_train()





