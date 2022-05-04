# Multimodal Emotion Detection (Video + Speech)

## TRIGGER project 

### Requirements
```
pip install -r requirements_trigger.txt
```

### Data preprocessing
- Video preprocessing: split video into 3-second clips; sample the frames with fps=30; extract and process faces into 256*256 images; save face images into .jpg files
- There are three students in each frame. There are self-defined bounding boxes for the positions of students on the left, middle, and right. Currently the bounding boxes ([[x_min, y_min], [x_max, y_max]]) are:
	- left_bb = [[100,450],[300, 900]]
	- mid_bb = [[750,450],[1000, 850]]
	- right_bb = [[1550,450],[1800, 850]]
- Calculate IOU of each detected face bounding boxes with the position boxes, and choose the one with highest IOU for the left, middle, right positions. If there is no overlap between the position box and the face bounding boxes, save a black image for this position of this frame.
- Save the audio of the clip into .wav file

Run script `python trigger_data_preprocessing.py`. Remember to change the video directory and target directory in the main function. It can support processing both a batch of videos (video folder) and a single video file. Examples in `trigger_data_preprocessing.py`. <br>

Result precessed data directory structure:<br>

|-processed_data<br>
&emsp;|- videoname_1<br>
&emsp;&emsp;|- clip_0<br>
&emsp;&emsp;&emsp;|- audio_clip.wav<br>
&emsp;&emsp;&emsp;|- left_frames<br>
&emsp;&emsp;&emsp;&emsp;|- frame_0.jpg<br>
&emsp;&emsp;&emsp;&emsp;|- ...<br>
&emsp;&emsp;&emsp;&emsp;|- frame_89.jpg<br>
&emsp;&emsp;&emsp;|- mid_frames<br>
&emsp;&emsp;&emsp;&emsp;|- frame_0.jpg<br>
&emsp;&emsp;&emsp;&emsp;|- ...<br>
&emsp;&emsp;&emsp;&emsp;|- frame_89.jpg<br>
&emsp;&emsp;&emsp;|- right_frames<br>
&emsp;&emsp;&emsp;&emsp;|- frame_0.jpg<br>
&emsp;&emsp;&emsp;&emsp;|- ...<br>
&emsp;&emsp;&emsp;&emsp;|- frame_89.jpg<br>
&emsp;&emsp;|- ...<br>
&emsp;&emsp;|- clip_x<br>
&emsp;|- videoname_2<br>
&emsp;|- ...<br>
&emsp;|- videoname_x<br>

### Model inference 

Derive an emotion label for each 3-second video clip. Emotion labels are ['neutral', 'calm', 'happy', 'sad', 'angry', 'fear', 'disgust', 'surprise']. Labels are saved into csv files. <br>

Example scripts:
- Multimodal:
```
python trigger_model_eval.py \
--processed_data_dir /scratch/project_2005901/full_processed_data \
--result_csv_path /scratch/project_2005901/full_processed_data/multimodal_result.csv \
--modality multi \
--checkpoint_path ./outputs/multimodal_model/modelParams-2.pkl \
--print_step_size 10
```

- Video-only:
```
python trigger_model_eval.py \
--processed_data_dir /scratch/project_2005901/full_processed_data \
--result_csv_path /scratch/project_2005901/full_processed_data/video_result.csv \
--modality video \
--checkpoint_path ./outputs/EmoNetLSTMfinetune/modelParams-1.pkl \
--print_step_size 10
```

- Audio-only:
```
python trigger_model_eval.py \
--processed_data_dir /scratch/project_2005901/full_processed_data \
--result_csv_path /scratch/project_2005901/full_processed_data/audio_result.csv \
--modality audio \
--checkpoint_path ./speech_emotion_classification_with_pytorch/pretrained_models/cnn_transf_parallel_model.pt \
--print_step_size 10
```

- Result csv file:
```
video_session: video name (e.g. demosharp)
clip_idx: clip index (e.g. 11)
clip_name: name of the clip directory (e.g. clip_11)
clip_start_timestamp: the start timestamp of the video clip (e.g. 33)
clip_end_timestamp: the end timestamp of the video clip (e.g. 36)
student_pos: left/mid/right
audio_path: audio file path
frames_path: frame image directory
num_frames: number of images in the frame image directory
label_idxs: for multimodal and audio-only model, label indexes are {1:'neutral', 2:'calm', 3:'happy', 4:'sad', 5:'angry', 6:'fear', 7:'disgust', 0:'surprise'}; for video-only model, label indexes are {0:'neutral', 1:'happy', 2:'sad', 3:'surprise', 4:'fear', 5:'disgust', 6:'angry', 7:'calm'}
labels: ['neutral', 'calm', 'happy', 'sad', 'angry', 'fear', 'disgust', 'surprise']
```

## Models trained on RAVDESS

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


