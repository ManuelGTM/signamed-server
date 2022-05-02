import argparse
import os
import shutil
import pprint

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.nn.functional as f
import torchvision.transforms as transforms
from collections import OrderedDict
import datetime

from PIL import Image
import numpy as np
import cv2

import cv2
import mediapipe as mp
import os, sys
import shutil
import datetime

from preprocessing.src.pose_hrnet import get_pose_net
from preprocessing.src.utils import pose_process, plot_pose
from preprocessing.config import cfg
from preprocessing.config import update_config

mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)

index_mirror = np.concatenate([
                [1,3,2,5,4,7,6,9,8,11,10,13,12,15,14,17,16],
                [21,22,23,18,19,20],
                np.arange(40,23,-1), np.arange(50,40,-1),
                np.arange(51,55), np.arange(59,54,-1),
                [69,68,67,66,71,70], [63,62,61,60,65,64],
                np.arange(78,71,-1), np.arange(83,78,-1),
                [88,87,86,85,84,91,90,89],
                np.arange(113,134), np.arange(92,113)
                ]) - 1
assert(index_mirror.shape[0] == 133)
#multi_scales = [512,640]
multi_scales = [512]

#MAX_FRAMES = 157
MAX_FRAMES = 200000


class GenKeypointsMediapipe():
    def __init__(self):
        self.holistic = mp_holistic.Holistic(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5, smooth_landmarks=True, model_complexity=2)
        
    def genKeypoints(self, video_path, output_img=None):
        filename = os.path.splitext(os.path.basename(video_path))[0]
        cap = cv2.VideoCapture(video_path)
        count = 0
        
        print ('genKeypoints: {}'.format(video_path))
        
        output_list = []
        count = 0
        while cap.isOpened():
            count = count + 1
            success, image = cap.read()
            if not success:
                break
            if count%1 != 0:
                continue
            image = cv2.flip(image, 1)
            image_temp = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image_temp.flags.writeable = False
            results = self.holistic.process(image_temp)

            idx_kps_pose = [0, 2, 5, 7, 8, 11, 12, 13, 14, 15, 16]
            kps_frame = np.zeros((53, 3))
            for idx, idx_mediapipe in enumerate(idx_kps_pose):
                if results.pose_landmarks:
                    kps_frame[idx][0] = results.pose_landmarks.landmark[idx_mediapipe].x
                    kps_frame[idx][1] = results.pose_landmarks.landmark[idx_mediapipe].y
                    kps_frame[idx][2] = results.pose_landmarks.landmark[idx_mediapipe].visibility
        
            
            for idx in range(21):
                if results.left_hand_landmarks:
                    kps_frame[idx+len(idx_kps_pose)][0] = results.left_hand_landmarks.landmark[idx].x
                    kps_frame[idx+len(idx_kps_pose)][1] = results.left_hand_landmarks.landmark[idx].y
                    kps_frame[idx+len(idx_kps_pose)][2] = 1
                    
                if results.right_hand_landmarks:
                    kps_frame[idx+len(idx_kps_pose)+21][0] = results.right_hand_landmarks.landmark[idx].x
                    kps_frame[idx+len(idx_kps_pose)+21][1] = results.right_hand_landmarks.landmark[idx].y
                    kps_frame[idx+len(idx_kps_pose)+21][2] = 1
                    
            #print(kps_frame.shape)
            output_list.append(kps_frame)
        
        output_list = np.array(output_list)
        print ('Read nº frames: {}'.format(output_list.shape))
        

        if (output_list.shape[0] > MAX_FRAMES):
            print ('Recortar el final, pues excede el maximo')
            output_list = output_list[0:MAX_FRAMES]

        return output_list


class GenKeypointsMediapipeC4():
    def __init__(self):
        self.holistic = mp_holistic.Holistic(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5, smooth_landmarks=True, model_complexity=2)
        
    def genKeypoints(self, video_path, output_img=None):
        filename = os.path.splitext(os.path.basename(video_path))[0]
        cap = cv2.VideoCapture(video_path)
        count = 0
        
        print ('genKeypoints: {}'.format(video_path))
        
        output_list = []
        count = 0
        while cap.isOpened():
            count = count + 1
            success, image = cap.read()
            if not success:
                break
            if count%1 != 0:
                continue
            image = cv2.flip(image, 1)
            image_temp = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image_temp.flags.writeable = False
            results = self.holistic.process(image_temp)

            idx_kps_pose = [0, 2, 5, 7, 8, 11, 12, 13, 14, 15, 16]
            kps_frame = np.zeros((53, 4))

            #idx_kps_pose = [0, 2, 5, 7, 8, 11, 12, 13, 14, 15, 16, 23, 24]
            #kps_frame = np.zeros((55, 4))

            for idx, idx_mediapipe in enumerate(idx_kps_pose):
                if results.pose_landmarks:
                    kps_frame[idx][0] = results.pose_landmarks.landmark[idx_mediapipe].x
                    kps_frame[idx][1] = results.pose_landmarks.landmark[idx_mediapipe].y
                    kps_frame[idx][2] = results.pose_landmarks.landmark[idx_mediapipe].z
                    kps_frame[idx][3] = results.pose_landmarks.landmark[idx_mediapipe].visibility
        
            
            for idx in range(21):
                if results.left_hand_landmarks:
                    kps_frame[idx+len(idx_kps_pose)][0] = results.left_hand_landmarks.landmark[idx].x
                    kps_frame[idx+len(idx_kps_pose)][1] = results.left_hand_landmarks.landmark[idx].y
                    kps_frame[idx+len(idx_kps_pose)][2] = results.left_hand_landmarks.landmark[idx].z
                    kps_frame[idx+len(idx_kps_pose)][3] = 1
                    
                if results.right_hand_landmarks:
                    kps_frame[idx+len(idx_kps_pose)+21][0] = results.right_hand_landmarks.landmark[idx].x
                    kps_frame[idx+len(idx_kps_pose)+21][1] = results.right_hand_landmarks.landmark[idx].y
                    kps_frame[idx+len(idx_kps_pose)+21][2] = results.right_hand_landmarks.landmark[idx].z
                    kps_frame[idx+len(idx_kps_pose)+21][3] = 1
                    
            #print(kps_frame.shape)
            output_list.append(kps_frame)
        
        output_list = np.array(output_list)
        print ('Read nº frames: {}'.format(output_list.shape))
        

        if (output_list.shape[0] > MAX_FRAMES):
            print ('Recortar el final, pues excede el maximo')
            output_list = output_list[0:MAX_FRAMES]

        return output_list

            

class GenKeypointsMMpose():
    def __init__(self, config_file, checkpoint_model, device=0):
        #torch.cuda.set_device(device)
        print ('init MMPose')
        cfg.merge_from_file(config_file)
        self.newmodel = get_pose_net(cfg, is_train=False)
        #print(self.newmodel)
        checkpoint = torch.load(checkpoint_model)
        
        state_dict = checkpoint['state_dict']
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            if 'backbone.' in k:
                name = k[9:] # remove module.
            if 'keypoint_head.' in k:
                name = k[14:] # remove module.
            new_state_dict[name] = v
        self.newmodel.load_state_dict(new_state_dict)

        self.newmodel.cuda().eval()
        self.transform  = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
    
    def norm_numpy_totensor(self, img):
        img = img.astype(np.float32) / 255.0
        for i in range(3):
            img[:, :, :, i] = (img[:, :, :, i] - mean[i]) / std[i]
        return torch.from_numpy(img).permute(0, 3, 1, 2)
    
    def stack_flip(self, img):
        img_flip = cv2.flip(img, 1)
        return np.stack([img, img_flip], axis=0)

    def merge_hm(self, hms_list):
        assert isinstance(hms_list, list)
        for hms in hms_list:
            hms[1,:,:,:] = torch.flip(hms[1,index_mirror,:,:], [2])
        
        hm = torch.cat(hms_list, dim=0)
        hm = torch.mean(hms, dim=0)
        return hm
    
    def genKeypoints(self, video_path, output_img=None):
        
        with torch.no_grad():
            cap = cv2.VideoCapture(video_path)
            frame_width = int(cap.get(3))
            frame_height = int(cap.get(4))
            
            output_list = []

            #idx = 0
            while cap.isOpened():
                success, img = cap.read()
                if not success:
                    break
                frame_height, frame_width = img.shape[:2]
                img = cv2.flip(img, flipCode=1)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                out = []
                for scale in multi_scales:
                    img_temp = cv2.resize(img, (scale,scale))
                    img_temp = self.stack_flip(img_temp)
                    img_temp = self.norm_numpy_totensor(img_temp).cuda()
                    hms = self.newmodel(img_temp)
                    out.append(f.interpolate(hms, (frame_width // 4,frame_height // 4), mode='bilinear'))

                out = self.merge_hm(out)
                result = out.reshape((133,-1))
                result = torch.argmax(result, dim=1)
                result = result.cpu().numpy().squeeze()
                y = result // (frame_width // 4)
                x = result % (frame_height // 4)
                pred = np.zeros((133, 3), dtype=np.float32)
                pred[:, 0] = x
                pred[:, 1] = y

                hm = out.cpu().numpy().reshape((133, frame_width//4, frame_height//4))
                pred = pose_process(pred, hm)
                pred[:,:2] *= 4.0 
                assert pred.shape == (133, 3)

                output_list.append(pred)
                # img = np.asarray(img)
                # img = plot_pose(img, pred)
                # cv2.imwrite('{}_{}.png'.format(output_img,idx), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
                # idx=idx+1
            
            output_list = np.array(output_list)
            cap.release()
            
        return output_list
            
            
class GenKeypointsOpenpose():
    def __init__(self, config_file, checkpoint_model):
        print ('init GenKeypointsOpenpose')
        
    def genKeypoints(self, video_path, output_json):
        print ('genKeypoints')
        gpu = '0'
        #openpose_bin_path_and_configuration = 'CUDA_VISIBLE_DEVICES='+gpu+' /home/gts/projects/mvazquez/openpose/build/examples/openpose/openpose.bin --net_resolution "720x720" --display 0 --scale_number 4 --scale_gap 0.25 --hand --hand_scale_number 6 --hand_scale_range 0.4 --render-pose=0 --model_folder /home/gts/projects/mvazquez/openpose/models'
        openpose_bin_path_and_configuration = 'CUDA_VISIBLE_DEVICES='+gpu+' /home/gts/projects/mvazquez/openpose/build/examples/openpose/openpose.bin --display 0 --render-pose=0 --hand --model_folder /home/gts/projects/mvazquez/openpose/models'
        command = openpose_bin_path_and_configuration+" --video "+video_path+" --write_json "+output_json+"/"
        
        if os.path.exists(output_json):
            print ("Remove folder: "+output_json) 
            shutil.rmtree(output_json)
            
        print ("Create folder: "+output_json) 
        os.makedirs(output_json)
        
        print(command)
        os.system(command)
        
        print ('end')
        