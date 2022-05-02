import os
import argparse
import shutil

def main(args):
    path_folder_work_dir = 'work_dir'
    path_folder_pretrained_models = 'pretrained-models'

    path = args.path
    weight = args.weight

    path_src = os.path.join(path_folder_work_dir,os.path.basename(path),'weights','weights-{}.pt'.format(weight))
    path_dest = os.path.join(path_folder_pretrained_models,os.path.basename(path)+'.pt')

    shutil.copy(path_src, path_dest)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', required=True, type=str)
    parser.add_argument('--weight', required=True, type=int)
    arg = parser.parse_args()
    main(arg)


## python save_checkpoint_in_pretrained_folder.py --path train_bone_motion_mediapipe-C3-xyc-version2-train1 --weight 75
