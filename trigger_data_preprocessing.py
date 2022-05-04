from moviepy.editor import *
import librosa
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from emonet.emonet.data_augmentation import DataAugmentor
import torch
import math
from torchvision import transforms


def bb_overlap(bb1, bb2):
    assert bb1[0][0] < bb1[1][0]
    assert bb1[0][1] < bb1[1][1]
    assert bb2[0][0] < bb2[1][0]
    assert bb2[0][1] < bb2[1][1]
    x_left = max(bb1[0][0], bb2[0][0])
    y_top = max(bb1[0][1], bb2[0][1])
    x_right = min(bb1[1][0], bb2[1][0])
    y_bottom = min(bb1[1][1], bb2[1][1])

    if x_right < x_left or y_bottom < y_top:
        return 0.0
    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    bb1_area = (bb1[1][0] - bb1[0][0]) * (bb1[1][1] - bb1[0][1])
    bb2_area = (bb2[1][0] - bb2[0][0]) * (bb2[1][1] - bb2[0][1])
    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    assert iou >= 0.0
    assert iou <= 1.0
    return iou
    
class DataPreprocessing():
    def __init__(self,
                clip_duration = 3,
                video_sample_rate = 30,
                audio_sample_rate = 48000,
                haarcascade_model = "haarcascade_frontalface_alt2.xml",
                LBFmodel = "lbfmodel.yaml",
                image_size = 256):
        self.clip_duration = clip_duration
        self.video_sample_rate = video_sample_rate
        self.audio_sample_rate = audio_sample_rate
        self.haarcascade_model = haarcascade_model
        self.face_detector = cv2.CascadeClassifier(haarcascade_model)
        self.LBFmodel = LBFmodel
        self.landmark_detector  = cv2.face.createFacemarkLBF()
        self.landmark_detector.loadModel(LBFmodel)
        self.left_bb = [[100,450],[300, 900]] #[[x_min, y_min], [x_max, y_max]]
        self.mid_bb = [[750,450],[1000, 850]]
        self.right_bb = [[1550,450],[1800, 850]]
        self.image_size = image_size
        self.transform_image_shape_no_flip = DataAugmentor(image_size, image_size)
        self.blank_img = np.zeros((image_size,image_size,3))
    
    def process_frame(self, image):
        H, W, C = image.shape
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.face_detector.detectMultiScale(image_gray)
        
        left_overlap = []
        mid_overlap = []
        right_overlap = []
        for face in faces:
            (x,y,w,d) = face
            left_overlap.append(bb_overlap([[x,y],[x+w,y+d]], self.left_bb))
            mid_overlap.append(bb_overlap([[x,y],[x+w,y+d]], self.mid_bb))
            right_overlap.append(bb_overlap([[x,y],[x+w,y+d]], self.right_bb))
        if max(left_overlap)==0:
            left_face = [0, 0, W, H]
        else:
            left_face = faces[left_overlap.index(max(left_overlap))]
        if max(mid_overlap)==0:
            mid_face = [0, 0, W, H]
        else:
            mid_face = faces[mid_overlap.index(max(mid_overlap))]
        if max(right_overlap)==0:
            right_face = [0, 0, W, H]
        else:
            right_face = faces[right_overlap.index(max(right_overlap))]
        faces = np.array([left_face, mid_face, right_face])
        
        _, landmarks = self.landmark_detector.fit(image_gray, faces)
        landmarks = np.array(landmarks).squeeze()
        assert len(faces)==len(landmarks)
#         print("landmarks.shape: ", landmarks.shape) #(3, 68, 2)
        imgs = []
        for i, landmark in enumerate(landmarks):
            if faces[i][0]==0 and faces[i][1]==0:
                img = self.blank_img.copy()
            else:
                landmark_bb = [landmark.min(axis=0)[0], landmark.min(axis=0)[1],landmark.max(axis=0)[0], landmark.max(axis=0)[1]]
                img, _ = self.transform_image_shape_no_flip(image, bb=landmark_bb)
            imgs.append(img)
        return imgs


    def preprocess_video_clip(self, video_clip):
        clip_frames = list(video_clip.iter_frames(fps=self.video_sample_rate))
#         print("frame shape: ", clip_frames[0].shape) #(1080, 1920, 3)
        left_imgs, mid_imgs, right_imgs = [], [], []
        for frame in clip_frames:
            left_img, mid_img, right_img = self.process_frame(frame)
            left_imgs.append(left_img)
            mid_imgs.append(mid_img)
            right_imgs.append(right_img)
        return left_imgs, mid_imgs, right_imgs

    def preprocess_single_video(self, video_path, video_clip_dir):
        full_video = VideoFileClip(video_path)
        total_len = int(full_video.duration)

        for clip_idx in range(int(total_len//self.clip_duration)):
            clip_start = clip_idx*self.clip_duration
            clip_end = min((clip_idx+1)*self.clip_duration, total_len)
            print("\t Processing clip {}-{}/{}".format(clip_start, clip_end, total_len))
            clip = full_video.subclip(clip_start, clip_end)
            clip_dir = os.path.join(video_clip_dir, 'clip_{}'.format(clip_idx))
            if not os.path.isdir(clip_dir):
                os.makedirs(clip_dir)
            audioclip = clip.audio
            audioclip.write_audiofile(os.path.join(clip_dir, 'audio_clip.wav'), fps=self.audio_sample_rate)
            left_imgs, mid_imgs, right_imgs = self.preprocess_video_clip(clip)
            left_imgs_dir = os.path.join(clip_dir, 'left_frames')
            if not os.path.isdir(left_imgs_dir):
                os.makedirs(left_imgs_dir)
            for img_idx, img in enumerate(left_imgs):
                cv2.imwrite(os.path.join(left_imgs_dir, 'frame_{}.jpg'.format(img_idx)), img)
            mid_imgs_dir = os.path.join(clip_dir, 'mid_frames')
            if not os.path.isdir(mid_imgs_dir):
                os.makedirs(mid_imgs_dir)
            for img_idx, img in enumerate(mid_imgs):
                cv2.imwrite(os.path.join(mid_imgs_dir, 'frame_{}.jpg'.format(img_idx)), img)
            right_imgs_dir = os.path.join(clip_dir, 'right_frames')
            if not os.path.isdir(right_imgs_dir):
                os.makedirs(right_imgs_dir)
            for img_idx, img in enumerate(right_imgs):
                cv2.imwrite(os.path.join(right_imgs_dir, 'frame_{}.jpg'.format(img_idx)), img)

    def preprocess(self, original_video_dir, processed_data_dir):
        if not os.path.isdir(processed_data_dir):
            os.makedirs(processed_data_dir)
            
        for video_file in os.listdir(original_video_dir):
            print("Preprocessing video {}".format(video_file))
            video_path = os.path.join(original_video_dir, video_file)
            video_clip_dir = os.path.join(processed_data_dir, video_file.split('.')[0])
            if not os.path.isdir(video_clip_dir):
                os.makedirs(video_clip_dir)
            self.preprocess_single_video(video_path, video_clip_dir)


if __name__ == "__main__":
    # Example command for preprocessing videos in a folder
    original_video_dir = '../original_videos'
    processed_data_dir = '../processed_data'
    if not os.path.isdir(processed_data_dir):
        os.makedirs(processed_data_dir)
    data_preprocessor = DataPreprocessing()
    data_preprocessor.preprocess(original_video_dir, processed_data_dir)

    # Example command for preprocessing a single video
    single_original_video_path = '/scratch/project_2005901/d1g1.mpeg'
    single_processed_data_dir = '../full_processed_data/d1g1'
    # os.makedirs('../full_processed_data')
    # os.makedirs('../full_processed_data/d1g1')
    data_preprocessor = DataPreprocessing()
    data_preprocessor.preprocess_single_video(single_original_video_path, single_processed_data_dir)

