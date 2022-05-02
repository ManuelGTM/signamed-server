import numpy as np
import json
import os
from enum import Enum

openpose = {
    'num_joint' : 55,
    'max_frame' : 300,
    'num_person' : 1,
    'channels' : 3,
    'thld_score' : 0,
    'num_person' : 1,
    'size_original' : 512,
    
    'body_pose_include' :[
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
        13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33,
        34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54
        ],
    'bone_pairs':(
            (1, 1),(2, 1),(3, 2),(4, 3),(5, 1),(6, 5),(7, 6),(8, 1),(0, 1),(9, 0),(10, 0),(11, 9),(12, 10),
            (13, 13),(13, 14),(14, 15),(15, 16),(16, 17),(13, 18),(18, 19),(19, 20),(20, 21),(13, 22),(22, 23),(23, 24),(24, 25),(13, 26),(26, 27),(27, 28),(28, 29),(13, 30),(30, 31),(31, 32),(32, 33),
            (34, 34),(34, 35),(35, 36),(36, 37),(37, 38),(34, 39),(39, 40),(40, 41),(41, 42),(34, 43),(43, 44),(44, 45),(45, 46),(34, 47),(47, 48),(48, 49),(49, 50),(34, 51),(51, 52),(52, 53),(53, 54)
        )
}

mmpose = {
    'num_joint' : 51,
    'max_frame' : 157,
    'num_person' : 1,
    'channels' : 3,
    'thld_score' : 0.1,
    'num_person' : 1,
    'size_original' : 512,
    
    'body_pose_include' :[
        0, 1, 2,  # face: noise & eyes
        5, 6,     # shoulders
        7, 8,     # elbows      
        9, 10,    # wrists
        91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111,  # left hand
        112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132 # right hand
        ],
    'bone_pairs':(
            (0, 0),(0, 1),(0, 2),(0, 3),(0, 4),(3, 5),(4, 6),(5, 7),(6, 8),
            (9, 9),(9, 10),(10, 11),(11, 12),(12, 13),(9, 14),(14, 15),(15, 16),(16, 17),(9, 18),(18, 19),(19, 20),(20, 21),(9, 22),(22, 23),(23, 24),(24, 25),(9, 26),(26, 27),(27, 28),(28, 29),
            (30, 30),(30, 31),(31, 32),(32, 33),(33, 34),(30, 35),(35, 36),(36, 37),(37, 38),(30, 39),(39, 40),(40, 31),(41, 42),(30, 43),(43, 44),(44, 45),(45, 46),(30, 47),(47, 48),(48, 49),(49, 50)
        )
}

mediapipe = {
    'num_joint' : 53,
    'max_frame' : 120,
    # 'max_frame' : 80000,
    'num_person' : 1,
    'thld_score' : 0.1,
    'num_person' : 1,
    
    'body_pose_include' :[
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, # pose
        11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,  # left hand
        32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52  # left hand
        ],
    'bone_pairs':(
        (0,0),(0,1),(0,2),(1,3),(2, 4),(0,5),(0,6),(5,7),(6,8),(7,9),(8,10),
        (11,11),(11,12),(12,13),(13,14),(14,15),(11,16),(16,17),(17,18),(18,19),(11,20),(20,21),(21,22),(22,23),(11,24),(24,25),(25,26),(26,27),(11,28),(28,29),(29,30),(30,31),
        (32,32),(32,33),(33,34),(34,35),(35,36),(32,37),(37,38),(38,39),(39,40),(30,41),(41,42),(42,43),(43,44),(32,45),(45,46),(46,47),(47,48),(32,49),(49,50),(50,51),(51,52)
        )
}

class MediapipeOptions(Enum):
    XYC = 1
    XYZ = 2
    XYZC = 3


def cut_kps_array_from_start(data, max_frame):
    data = data[0:max_frame,:,:]
    return data

def cut_kps_array_from_middle(data, number_frames_video, max_frame):
    middle_number_frames_video = number_frames_video / 2
    start = int(middle_number_frames_video - (max_frame/2))
    end = start + max_frame

    data = data[start:end,:,:]
    return data

def adjust_Z_hands(data):
    data_change = np.copy(data)
    for i_frame in range(data.shape[0]):
        data_change[i_frame,11:32,2] = data_change[i_frame,11:32,2]+data_change[i_frame,10,2]
        data_change[i_frame,32:53,2] = data_change[i_frame,32:53,2]+data_change[i_frame,9,2]
    return data_change


class GenFeaturesMediapipeC3():
    def __init__(self):
        print ('GenFeaturesMediapipeC3')
        print ('Number joints: {}'.format(mediapipe['num_joint']))
        
    def getFeatures(self, data):
        channels = 3
        number_frames_video = data.shape[0]
        if number_frames_video > mediapipe['max_frame']:
            data = cut_kps_array_from_middle(data, number_frames_video, mediapipe['max_frame'])

        data_joints = np.zeros((channels, mediapipe['max_frame'], mediapipe['num_joint'], mediapipe['num_person']))
        for idx, frame in enumerate(data):
            pose = [v for i, v in enumerate(frame.flatten()) if i % 3 != 2 and i // 3 in mediapipe['body_pose_include']]
            score = [v for i, v in enumerate(frame.flatten()) if i % 3 == 2 and i // 3 in mediapipe['body_pose_include']]
            for m in range(mediapipe['num_person']):
                data_joints[0, idx, :, m] = pose[0::2]
                data_joints[1, idx, :, m] = pose[1::2]
                data_joints[2, idx, :, m] = score

        data_joints[0:2] = data_joints[0:2] - 0.5
        #data_joints[1:2] = -data_joints[1:2]
        data_joints[2][data_joints[2] < mediapipe['thld_score']] = 0
        data_joints[0][data_joints[2] == 0] = 0
        data_joints[1][data_joints[2] == 0] = 0
        
        data_bones = np.zeros((channels, mediapipe['max_frame'], mediapipe['num_joint'], mediapipe['num_person']))
        for v1, v2 in mediapipe['bone_pairs']:
            v1 -= 1
            v2 -= 1
            data_bones[:, :, v1, :] = data_joints[:, :, v1, :] - data_joints[:, :, v2, :]

        data_motion_joints = np.zeros((channels, mediapipe['max_frame'], mediapipe['num_joint'], mediapipe['num_person']))
        data_motion_bones = np.zeros((channels, mediapipe['max_frame'], mediapipe['num_joint'], mediapipe['num_person']))
        for t in range(mediapipe['max_frame'] - 1): 
            data_motion_joints[:, t, :, :] = data_joints[:, t + 1, :, :] - data_joints[:, t, :, :]
            data_motion_bones[:, t, :, :] = data_bones[:, t + 1, :, :] - data_bones[:, t, :, :]
        data_motion_joints[:, mediapipe['max_frame'] - 1, :, :] = 0
        data_motion_bones[:, mediapipe['max_frame'] - 1, :, :] = 0
            
        return data_joints, data_bones, data_motion_joints, data_motion_bones


class GenFeaturesMediapipeC4():
    def __init__(self, option, mscaleX = 1, adjustZ = False, noframeslimit = False):
        print ('GenFeaturesMediapipeC4')
        print ('Number joints: {}'.format(mediapipe['num_joint']))
        print ('Option: {}'.format(option))
        self.option = option
        self.mscaleX = mscaleX
        self.adjustZ = adjustZ
        self.noframeslimit = noframeslimit

    def getFeatures(self, data):
        channels = 4
        number_frames_video = data.shape[0]
        if (self.noframeslimit == False):
            if number_frames_video > mediapipe['max_frame']:
                data = cut_kps_array_from_middle(data, number_frames_video, mediapipe['max_frame'])
                number_frames_video = mediapipe['max_frame']

        if (self.adjustZ == True):
            data = adjust_Z_hands(data)

        number_frames_video = mediapipe['max_frame']
        data_joints = np.zeros((channels, number_frames_video, mediapipe['num_joint'], mediapipe['num_person']))
        for idx, frame in enumerate(data):
            # pose = [v for i, v in enumerate(frame.flatten()) if i % 3 != 2 and i // 3 in mediapipe['body_pose_include']]
            # score = [v for i, v in enumerate(frame.flatten()) if i % 3 == 2 and i // 3 in mediapipe['body_pose_include']]

            pose = [v for i, v in enumerate(frame.flatten()) if i % 4 != 3 and i // 4 in mediapipe['body_pose_include']]    
            score = [v for i, v in enumerate(frame.flatten()) if i % 4 == 3 and i // 4 in mediapipe['body_pose_include']]

            for m in range(mediapipe['num_person']):
                data_joints[0, idx, :, m] = pose[0::3]
                data_joints[1, idx, :, m] = pose[1::3]
                data_joints[2, idx, :, m] = pose[2::3]
                data_joints[3, idx, :, m] = score

        data_joints[0:3] = data_joints[0:3] - 0.5
        #data_joints[1:3] = -data_joints[1:3]
        #data_joints[2:3] = -data_joints[2:3]
        data_joints[3][data_joints[3] < mediapipe['thld_score']] = 0
        data_joints[0][data_joints[3] == 0] = 0
        data_joints[1][data_joints[3] == 0] = 0
        data_joints[2][data_joints[3] == 0] = 0

        # factor = 4/3  # Quitar los lados. para dejar aspect ratio 1:1
        # factor = 1  # Quitar los lados. para dejar aspect ratio 1:1
        data_joints[0, :, :, :] = data_joints[0, :, :, :] * self.mscaleX

        data_bones = np.zeros((channels, number_frames_video, mediapipe['num_joint'], mediapipe['num_person']))
        for v1, v2 in mediapipe['bone_pairs']:
            v1 -= 1
            v2 -= 1
            data_bones[:, :, v1, :] = data_joints[:, :, v1, :] - data_joints[:, :, v2, :]

        data_motion_joints = np.zeros((channels, number_frames_video, mediapipe['num_joint'], mediapipe['num_person']))
        data_motion_bones = np.zeros((channels, number_frames_video, mediapipe['num_joint'], mediapipe['num_person']))
        for t in range(number_frames_video - 1): 
            data_motion_joints[:, t, :, :] = data_joints[:, t + 1, :, :] - data_joints[:, t, :, :]
            data_motion_bones[:, t, :, :] = data_bones[:, t + 1, :, :] - data_bones[:, t, :, :]
        data_motion_joints[:, number_frames_video - 1, :, :] = 0
        data_motion_bones[:, number_frames_video - 1, :, :] = 0

        if (self.option==MediapipeOptions.XYC):
            data_joints = data_joints[[0,1,3],:,:,:]
            data_bones = data_bones[[0,1,3],:,:,:]
            data_motion_joints = data_motion_joints[[0,1,3],:,:,:]
            data_motion_bones = data_motion_bones[[0,1,3],:,:,:]
        elif (self.option==MediapipeOptions.XYZ):
            data_joints = data_joints[[0,1,2],:,:,:]
            data_bones = data_bones[[0,1,2],:,:,:]
            data_motion_joints = data_motion_joints[[0,1,2],:,:,:]
            data_motion_bones = data_motion_bones[[0,1,2],:,:,:]
            
        return data_joints, data_bones, data_motion_joints, data_motion_bones



class GenFeaturesMMPose():
    def __init__(self):
        print ('init')
        print ( mmpose['num_joint'])
        
    def getFeatures(self, data):
        data_joints = np.zeros((mmpose['channels'], mmpose['max_frame'], mmpose['num_joint'], mmpose['num_person']))
        for idx, frame in enumerate(data):
            pose = [(v/ mmpose['size_original']) for i, v in enumerate(frame.flatten()) if i % 3 != 2 and i // 3 in mmpose['body_pose_include']]
            score = [v for i, v in enumerate(frame.flatten()) if i % 3 == 2 and i // 3 in mmpose['body_pose_include']]
            for m in range(mmpose['num_person']):
                data_joints[0, idx, :, m] = pose[0::2]
                data_joints[1, idx, :, m] = pose[1::2]
                data_joints[2, idx, :, m] = score

        data_joints[0:2] = data_joints[0:2] - 0.5
        data_joints[1:2] = -data_joints[1:2]
        data_joints[2][data_joints[2] < mmpose['thld_score']] = 0
        data_joints[0][data_joints[2] == 0] = 0
        data_joints[1][data_joints[2] == 0] = 0
        
        data_bones = np.zeros((mmpose['channels'], mmpose['max_frame'], mmpose['num_joint'], mmpose['num_person']))
        for v1, v2 in mmpose['bone_pairs']:
            v1 -= 1
            v2 -= 1
            data_bones[:, :, v1, :] = data_joints[:, :, v1, :] - data_joints[:, :, v2, :]
            
        return data_joints, data_bones
    
    
class GenFeaturesOpenpose():
    def __init__(self):
        print ('init')
        print (openpose['max_frame'])
        
        print (openpose['num_joint'])
        
        
    def getFeatures(self, data_path):
        files = os.listdir(data_path)
        print ('len_files: {}'.format(len(files)))
        sample_id = files[0][:-28]
        
        data = []
        for i in range(len(files)):
            
            file = sample_id+'_'+(str(i).zfill(12))+'_keypoints.json'
            path_file =  os.path.join(data_path, file)
            with open(path_file) as f:
                data_json = json.load(f)
            pose_content = data_json["people"][0]
            body_pose = pose_content["pose_keypoints_2d"]
            left_hand_pose = pose_content["hand_left_keypoints_2d"]
            right_hand_pose = pose_content["hand_right_keypoints_2d"]
            body_pose.extend(left_hand_pose)
            body_pose.extend(right_hand_pose)  
            
            data.append([(v/openpose['size_original']) for i, v in enumerate(body_pose) if i // 3 in openpose['body_pose_include']]) 

        data_joints = np.zeros((openpose['channels'], openpose['max_frame'], openpose['num_joint'], openpose['num_person']))
        for idx, frame in enumerate(data):
            for m in range(openpose['num_person']):
                data_joints[0, idx, :, m] = frame[0::3]
                data_joints[1, idx, :, m] = frame[1::3]
                data_joints[2, idx, :, m] = frame[2::3]

        data_joints[0:2] = data_joints[0:2] - 0.5
        data_joints[1:2] = -data_joints[1:2]
        data_joints[2][data_joints[2] < openpose['thld_score']] = 0
        data_joints[0][data_joints[2] == 0] = 0
        data_joints[1][data_joints[2] == 0] = 0
        
        data_bones = np.zeros((openpose['channels'], openpose['max_frame'], openpose['num_joint'], openpose['num_person']))
        for v1, v2 in openpose['bone_pairs']:
            v1 -= 1
            v2 -= 1
            data_bones[:, :, v1, :] = data_joints[:, :, v1, :] - data_joints[:, :, v2, :]
            
        return data_joints, data_bones
        
        
    