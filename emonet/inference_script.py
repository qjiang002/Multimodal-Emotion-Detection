import numpy as np
from pathlib import Path
import argparse

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data.sampler import WeightedRandomSampler
from torchvision import transforms 

from emonet.models import EmoNet
from emonet.data import AffectNet
from emonet.data_augmentation import DataAugmentor
from emonet.metrics import CCC, PCC, RMSE, SAGR, ACC
from emonet.evaluation import evaluate, evaluate_flip
from PIL import Image
from skimage import io
from skimage.transform import rescale, resize, downscale_local_mean


image_size = 256
n_expression = 8
device = 'cuda' if torch.cuda.is_available() else 'cpu'


img_path = '../ravdess-emotional-speech-audio/videos/Actor_01/01-01-01-01-01-01-01/frame18.jpg'
# image = Image.open(img_path)
image = io.imread(img_path)
s = min(image.shape[0], image.shape[1])
image = resize(image, (s, s), anti_aliasing=True)
print("initial image: ", type(image))
print(image.shape, image.dtype)

transform_image = transforms.Compose([transforms.ToTensor()])
transform_image_shape_no_flip = DataAugmentor(image_size, image_size)

image, landmarks = transform_image_shape_no_flip(image)
image = transform_image(image)
input_tensor = torch.unsqueeze(image, 0).type(torch.FloatTensor).to(device)
print("processed input_tensor: ", type(input_tensor), input_tensor.shape)



# Loading the model 
state_dict_path = Path(__file__).parent.joinpath('pretrained', f'emonet_{n_expression}.pth')

print(f'Loading the model from {state_dict_path}.')
state_dict = torch.load(str(state_dict_path), map_location='cpu')
state_dict = {k.replace('module.',''):v for k,v in state_dict.items()}
net = EmoNet(n_expression=n_expression).to(device)
net.load_state_dict(state_dict, strict=False)
net.eval()

output = net(input_tensor)

print(output)

