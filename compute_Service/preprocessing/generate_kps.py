import numpy as np
import os, sys
import tqdm
import argparse
import shutil

sys.path.append('..')
from preprocessing.src.gen_keypoints import GenKeypointsMediapipeC4 as Keypoints


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

    gen_keypoints = Keypoints()
    folder_videos = args.input
    folder_out = args.output
    print ('LIST_VIDEOS')
    list_videos = os.listdir(folder_videos)
    print (list_videos)
    print (len(list_videos))

    create_folder(folder_out)

    
    for video in tqdm.tqdm(list_videos):
        #video = list_videos[0]
        #print ('process : {}'.format(video))    
        filename = os.path.splitext(os.path.basename(video))[0]
        video_path = os.path.join(folder_videos, video)
        out_path = os.path.join(folder_out, filename+'.npy')
        keypoints = gen_keypoints.genKeypoints(video_path)
        np.save(out_path, keypoints)


if __name__ == '__main__':
    #config = '/home/gts/projects/mvazquez/wholepose/wholebody_w48_384x288.yaml'
    #checkpoint_model = '/home/gts/projects/mvazquez/wholepose/hrnet_w48_coco_wholebody_384x288-6e061c6a_20200922.pth'

    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, type=str)
    parser.add_argument('--output', required=True, default='', type=str)
    #arg = parser.parse_args([
    #    '--input', "/home/temporal2/mvazquez/bdd/SignaMed/SESIONES_REV_ELAN_PROCESSED/data",
    #    '--output', "/home/temporal2/mvazquez/bdd/SignaMed/Signamed_mediapipe_Complex2/npy/kps",
    #    ])

    arg = parser.parse_args()
    main(arg)

    """
    python generate_kps.py --input /almacen/bdd/LSE_Lex40_uvigo/SignaMed/data --output /home/temporal2/mvazquez/bdd/SignaMed/Signamed_mediapipe_Complex/npy/kps-remove

    python generate_kps.py --input /almacen/bdd/LSE_Lex40_uvigo/SignaMed/data --output /home/temporal2/mvazquez/bdd/SignaMed/Signamed_mediapipe_Complex2/npy/kps

    """
    