import numpy as np
import os, sys
import tqdm
import argparse
import shutil

sys.path.append('..')
from preprocessing.src.gen_features import MediapipeOptions
from preprocessing.src.gen_features import GenFeaturesMediapipeC4 as Features

def create_folder(folder):
    print ('create_folder: {}'.format(folder))    
    try:
        os.makedirs(folder)    
        print("Directory " , folder ,  " Created ")
    except FileExistsError:
        print("Directory " , folder ,  " already exists")
        shutil.rmtree(folder, ignore_errors=True)
        os.makedirs(folder) 
        print("Directory " , folder ,  " reset")


def main(args):
    folder = arg.folder
    extra = arg.extra
    mscaleX = arg.mscaleX
    adjustZ = arg.adjustZ
    noframeslimit = arg.noframeslimit
    list_videos = os.listdir(os.path.join(folder, 'kps'))
    #print (list_videos)
    print (len(list_videos))

    folder_out_joints = os.path.join(folder, 'joints_'+extra)
    folder_out_bones = os.path.join(folder, 'bones_'+extra)
    folder_out_joints_motion = os.path.join(folder, 'joints_motion_'+extra)
    folder_out_bones_motion = os.path.join(folder, 'bones_motion_'+extra)

    create_folder(folder_out_joints)
    create_folder(folder_out_bones)
    create_folder(folder_out_joints_motion)
    create_folder(folder_out_bones_motion)

    if (extra=='C3_xyc'):
        genFeatures = Features(MediapipeOptions.XYC, mscaleX, adjustZ, noframeslimit)
    elif (extra=='C3_xyz'):
        genFeatures = Features(MediapipeOptions.XYZ, mscaleX, adjustZ, noframeslimit)
    elif (extra=='C4_xyzc'):
        genFeatures = Features(MediapipeOptions.XYZC, mscaleX, adjustZ, noframeslimit)

    
    for video in tqdm.tqdm(list_videos):
        filename = os.path.splitext(os.path.basename(video))[0]
        file_path = os.path.join(folder, 'kps', filename+'.npy')
        keypoints = np.load(file_path)


        data_joints, data_bones, data_joints_motion, data_bones_motion = genFeatures.getFeatures(keypoints)
        out_path = os.path.join(folder_out_joints, filename+'.npy')
        np.save(out_path, data_joints)
        out_path = os.path.join(folder_out_bones, filename+'.npy')
        np.save(out_path, data_bones)
        out_path = os.path.join(folder_out_joints_motion, filename+'.npy')
        np.save(out_path, data_joints_motion)
        out_path = os.path.join(folder_out_bones_motion, filename+'.npy')
        np.save(out_path, data_bones_motion)

        #data_joints, data_bones = genFeatures.genFeaturesMotion(keypoints)

if __name__ == '__main__':
    #config = '/home/gts/projects/mvazquez/wholepose/wholebody_w48_384x288.yaml'
    #checkpoint_model = '/home/gts/projects/mvazquez/wholepose/hrnet_w48_coco_wholebody_384x288-6e061c6a_20200922.pth'

    parser = argparse.ArgumentParser()
    parser.add_argument('--folder', required=True, type=str)
    parser.add_argument('--extra', required=True, default='', type=str)
    parser.add_argument('--mscaleX', required=False, default=1, type=float)
    parser.add_argument('--adjustZ', required=False, default=False, type=bool)
    parser.add_argument('--noframeslimit', required=False, default=False, type=bool)
    #arg = parser.parse_args([
    #    '--folder', "/home/temporal2/mvazquez/bdd/AUTSL_mediapipe_Complex2/npy",
    #    '--extra', "C3_xyc",
    #    ])
    arg = parser.parse_args()
    print (arg)
    main(arg)

    # --extra = C3_xyc
    # --extra = C3_xyz
    # --extra = C4_xyzc

    """
    python generate_features.py --folder /home/temporal2/mvazquez/bdd/SignaMed/Signamed_mediapipe_Complex2/version1/npy --extra C3_xyc
    python generate_features.py --folder /home/temporal2/mvazquez/bdd/SignaMed/Signamed_mediapipe_Complex2/version1/npy --extra C3_xyz
    python generate_features.py --folder /home/temporal2/mvazquez/bdd/SignaMed/Signamed_mediapipe_Complex2/version1/npy --extra C4_xyzc

    """


    