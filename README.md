# Multimodal Emotion Detection (Video + Speech)

### Requirements
```
pip install -r requirements.txt
```

### Dataset
Download [RAVDESS](https://zenodo.org/record/1188976#.YmyVnZPMK3I) dataset and [RAVDESS Facial Landmark](https://zenodo.org/record/3255102#.YmyYOJPMK3I) and put them in the following directory

|-ravdess_data<br>
&emsp;|- audio_speech_actors_01-24<br>
&emsp;&emsp;|- Actor_01<br>
&emsp;&emsp;|- ...<br>
&emsp;&emsp;|- Actor_24<br>
&emsp;|- facial_landmarks<br>
&emsp;&emsp;|- 01-01-01-01-01-01-01.csv<br>
&emsp;&emsp;|- ...<br>
&emsp;|- videos<br>
&emsp;&emsp;|- Actor_01<br>
&emsp;&emsp;|- ...<br>
&emsp;&emsp;|- Actor_24<br>

### Data processing
Video data: each video segment is 3 seconds with sampling rate fps=30. Each frame with facial landmark labels is augmented with `emonet.emonet.data_augmentation.DataAugmentor` which takes in the bounding box of the face, centers the face and scales it into 256-by-256 image. The final input video data has shape (90, 3, 256, 256). <br>

Speech data: each speech segment is also 3 seconds and is sampled with `librosa.load(speech_path, duration=3, offset=0.5, sr=SAMPLE_RATE)` with `SAMPLE_RATE = 48000`. It is then augmented with Gaussian noise as in `ravdess_dataset.addAWGN` and transformed into MELspectrogram as in `ravdess_dataset.getMELspectrogram`. The final input speech data has shape (128, 563). <br>

### Speech-only model
The speech-only model is the pre-trained Parallel 2D CNN - Transformer Encoder model from the repo [Speech-Emotion-Classification-with-PyTorch](https://github.com/Data-Science-kosta/Speech-Emotion-Classification-with-PyTorch.git).<br>

Model directory: `./speech_emotion_classification_with_pytorch/models/cnn_transformer.py`
To evaluate the pre-trained speech model:
```
CUDA_VISIBLE_DEVICES=5 python speech_net_train_eval.py --val --load_checkpoint --checkpoint_path ./speech_emotion_classification_with_pytorch/pretrained_models/cnn_transf_parallel_model.pt
```
```
Eval accuracy = 0.87
```
Labels: `EMOTIONS = {1:'neutral', 2:'calm', 3:'happy', 4:'sad', 5:'angry', 6:'fear', 7:'disgust', 0:'surprise'}`

### EmonetLSTM (video-only) model
EmonetLSTM is based on [Emonet](https://github.com/face-analysis/emonet) which is an emotion recognition model trained on the image dataset AffectNet. To adapt Emonet to emotion videos, an LSTM layer is attached to the outputs of Emonet of the video frames.<br>

Model directory:`./emonet/emonet/models/emonet_lstm.py`
To evaluate the pre-trianed EmonetLSTM model:
```
CUDA_VISIBLE_DEVICES=5 python emonet_train_eval.py --val --load_checkpoint --checkpoint_path ./outputs/EmoNetLSTMfinetune/modelParams-1.pkl
```
```
./outputs/EmoNetLSTMfinetune/modelParams-1.pkl: Eval accuracy = 0.57
```
Labels: `{'neutral':0, 'happy':1, 'sad':2, 'surprise':3, 'fear':4, 'disgust':5, 'angry':6, 'calm':7}`

### Multimodal model
The multimodal model combines the pre-trained speech-only model and the pre-trained EmonetLSTM by concatenating their outputs and attaching a feedforward layer. <br>

Model directory:`./video_speech_model.py`
To evaluate the pre-trianed multimodal model:
```
CUDA_VISIBLE_DEVICES=5 python multimodal_train_eval.py --val --load_checkpoint --checkpoint_path ./outputs/multimodal_model/modelParams-2.pkl
```
```
./outputs/multimodal_model/modelParams-2.pkl: Eval accuracy = 0.83
./outputs/multimodal_model/modelParams-1.pkl: Eval accuracy = 0.75
```
Labels: `EMOTIONS = {1:'neutral', 2:'calm', 3:'happy', 4:'sad', 5:'angry', 6:'fear', 7:'disgust', 0:'surprise'}`


