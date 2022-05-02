import pickle
import numpy as np
import os, sys
import csv
import argparse
import shutil

def create_folder(folder):
    print ('create_folder: {}'.format(folder))    
    if not os.path.exists(folder):
        os.makedirs(folder)    
        print("Directory " , folder ,  " Created ")
    else:
        print("Directory " , folder ,  " already exists")
        shutil.rmtree(folder, ignore_errors=True)
        os.makedirs(folder) 
        print("Directory " , folder ,  " reset")

def load_Split_Labels(split_csv):
    split = []
    labels = []
    with open(split_csv) as csv_file:
        for row in csv.reader(csv_file, delimiter=','): # each row is a list
            split.append(row[0])
            labels.append(np.int64(row[1]))
    return split, labels

def generate_dataset_subset(split, labels, dataset_joints, dataset_bones,dataset_joints_motion, dataset_bones_motion,folder_in_joints, folder_in_bones, folder_in_joints_motion, folder_in_bones_motion):

    split_joints = []
    split_bones = []
    split_joints_motion = []
    split_bones_motion = []

    split_ok = []
    label_ok = []

    for idx, file in enumerate(split):
        file = file + '_color'  # autsl necesatario añadir _color
        try:
            arr_joints = np.load(os.path.join(folder_in_joints, file+'.npy'))
            arr_bones = np.load(os.path.join(folder_in_bones, file+'.npy'))

            arr_joints_motion = np.load(os.path.join(folder_in_joints_motion, file+'.npy'))
            arr_bones_motion = np.load(os.path.join(folder_in_bones_motion, file+'.npy'))
            
            dataset_joints.append(arr_joints)
            dataset_bones.append(arr_bones)
            split_joints.append(arr_joints)
            split_bones.append(arr_bones)

            dataset_joints_motion.append(arr_joints_motion)
            dataset_bones_motion.append(arr_bones_motion)
            split_joints_motion.append(arr_joints_motion)
            split_bones_motion.append(arr_bones_motion)
            
            split_ok.append(split[idx])
            label_ok.append(labels[idx])
        except:
            print ('discard: {}'.format(file))
    return dataset_joints, dataset_bones, dataset_joints_motion, dataset_bones_motion, split_joints, split_bones, split_joints_motion, split_bones_motion, split_ok, label_ok


def main(args):
    extra = args.extra
    folder_npy = args.folder_npy
    folder_out = args.folder_out
    folder_labels = args.folder_labels

    proccess_test = True
    process_test_movil = True
    process_test_collaborations = True

    if (extra!=''):
        folder_out = os.path.join(folder_out,extra)
        folder_in_joints = os.path.join(folder_npy,'joints'+'_'+extra)
        folder_in_bones = os.path.join(folder_npy,'bones'+'_'+extra)
        folder_in_joints_motion = os.path.join(folder_npy,'joints_motion'+'_'+extra)
        folder_in_bones_motion = os.path.join(folder_npy,'bones_motion'+'_'+extra)

    create_folder(folder_out)
    
    train_split_csv = os.path.join(folder_labels, 'train_labels.csv')
    val_split_csv = os.path.join(folder_labels, 'val_labels.csv')
    test_split_csv = os.path.join(folder_labels, 'test_labels.csv')
    test_movil_split_csv = os.path.join(folder_labels, 'test_movil_labels.csv')
    test_collaborations_split_csv = os.path.join(folder_labels, 'test_collaborations_labels.csv')

    #list_files = os.listdir(folder_in_joints)
    #files = [os.path.splitext(file)[0] for file in list_files]

    train_split, train_labels = load_Split_Labels(train_split_csv)
    val_split, val_labels = load_Split_Labels(val_split_csv)

    try:
        test_split, test_labels = load_Split_Labels(test_split_csv)
    except:
        proccess_test = False

    try:
        test_movil_split, test_movil_labels = load_Split_Labels(test_movil_split_csv)
    except:
        process_test_movil = False

    try:
        test_collaborations_split, test_collaborations_labels = load_Split_Labels(test_collaborations_split_csv)
    except:
        process_test_collaborations = False
    

    dataset_joints = []
    dataset_bones = []
    dataset_joints_motion = []
    dataset_bones_motion = []
    
    dataset_joints, dataset_bones, dataset_joints_motion, dataset_bones_motion, train_split_joints, train_split_bones, train_split_joints_motion, train_split_bones_motion, train_split_ok, train_labels_ok = generate_dataset_subset(train_split, train_labels, dataset_joints, dataset_bones, dataset_joints_motion, dataset_bones_motion, folder_in_joints, folder_in_bones, folder_in_joints_motion, folder_in_bones_motion)
    dataset_joints, dataset_bones, dataset_joints_motion, dataset_bones_motion, val_split_joints, val_split_bones, val_split_joints_motion, val_split_bones_motion, val_split_ok, val_labels_ok = generate_dataset_subset(val_split, val_labels, dataset_joints, dataset_bones, dataset_joints_motion, dataset_bones_motion, folder_in_joints, folder_in_bones, folder_in_joints_motion, folder_in_bones_motion)
    if proccess_test: 
        dataset_joints, dataset_bones, dataset_joints_motion, dataset_bones_motion, test_split_joints, test_split_bones, test_split_joints_motion, test_split_bones_motion, test_split_ok, test_labels_ok = generate_dataset_subset(test_split, test_labels, dataset_joints, dataset_bones, dataset_joints_motion, dataset_bones_motion, folder_in_joints, folder_in_bones, folder_in_joints_motion, folder_in_bones_motion)
    if process_test_movil:
        dataset_joints, dataset_bones, dataset_joints_motion, dataset_bones_motion, test_movil_split_joints, test_movil_split_bones, test_movil_split_joints_motion, test_movil_split_bones_motion, test_movil_split_ok, test_movil_labels_ok = generate_dataset_subset(test_movil_split, test_movil_labels, dataset_joints, dataset_bones, dataset_joints_motion, dataset_bones_motion, folder_in_joints, folder_in_bones, folder_in_joints_motion, folder_in_bones_motion)
    if process_test_collaborations:
        dataset_joints, dataset_bones, dataset_joints_motion, dataset_bones_motion, test_collaborations_split_joints, test_collaborations_split_bones, test_collaborations_split_joints_motion, test_collaborations_split_bones_motion, test_collaborations_split_ok, test_collaborations_labels_ok = generate_dataset_subset(test_collaborations_split, test_collaborations_labels, dataset_joints, dataset_bones, dataset_joints_motion, dataset_bones_motion, folder_in_joints, folder_in_bones, folder_in_joints_motion, folder_in_bones_motion)



    # print ('dataset_joints: {}'.format(len(dataset_joints)))
    # print ('dataset_bones: {}'.format(len(dataset_bones)))
    # print ('dataset_joints_motion: {}'.format(len(dataset_joints_motion)))
    # print ('dataset_bones_motion: {}'.format(len(dataset_bones_motion)))

    # print ('train_split_joints: {}'.format(len(train_split_joints)))
    # print ('val_split_joints: {}'.format(len(val_split_joints)))
    # print ('test_split_joints: {}'.format(len(test_split_joints)))
    # print ('test_movil_split_joints: {}'.format(len(test_movil_split_joints)))

    # print ('train_split_bones: {}'.format(len(train_split_bones)))
    # print ('val_split_bones: {}'.format(len(val_split_bones)))
    # print ('test_split_bones: {}'.format(len(test_split_bones)))
    # print ('test_movil_split_bones: {}'.format(len(test_movil_split_bones)))

    # print ('train_split_joints_motion: {}'.format(len(train_split_joints_motion)))
    # print ('val_split_joints_motion: {}'.format(len(val_split_joints_motion)))
    # print ('test_split_joints_motion: {}'.format(len(test_split_joints_motion)))
    # print ('test_movil_split_joints_motion: {}'.format(len(test_movil_split_joints_motion)))

    # print ('train_split_bones_motion: {}'.format(len(train_split_bones_motion)))
    # print ('val_split_bones_motion: {}'.format(len(val_split_bones_motion)))
    # print ('test_split_bones_motion: {}'.format(len(test_split_bones_motion)))
    # print ('test_movil_split_bones_motion: {}'.format(len(test_movil_split_bones_motion)))

    ## SAVE 
    ## TRAIN SET
    train_split_joints = np.array(train_split_joints)
    train_split_bones = np.array(train_split_bones)
    train_split_joints_motion = np.array(train_split_joints_motion)
    train_split_bones_motion = np.array(train_split_bones_motion)
    print ('train_data_joint: {}'.format(train_split_joints.shape))
    print ('train_data_bone: {}'.format(train_split_bones.shape))
    print ('train_split_joints_motion: {}'.format(train_split_joints_motion.shape))
    print ('train_split_bones_motion: {}'.format(train_split_bones_motion.shape))
    np.save(os.path.join(folder_out, 'train_data_joint.npy'), train_split_joints)
    np.save(os.path.join(folder_out, 'train_data_bone.npy'), train_split_bones)
    np.save(os.path.join(folder_out, 'train_data_joint_motion.npy'), train_split_joints_motion)
    np.save(os.path.join(folder_out, 'train_data_bone_motion.npy'), train_split_bones_motion)
    with open(os.path.join(folder_out, 'train_label.pkl'), 'wb') as f:
        pickle.dump((train_split_ok, train_labels_ok), f)

    ## VAL SET
    val_split_joints = np.array(val_split_joints)
    val_split_bones = np.array(val_split_bones)
    val_split_joints_motion = np.array(val_split_joints_motion)
    val_split_bones_motion = np.array(val_split_bones_motion)
    print ('val_data_joint: {}'.format(val_split_joints.shape))
    print ('val_data_bone: {}'.format(val_split_bones.shape))
    print ('val_split_joints_motion: {}'.format(val_split_joints_motion.shape))
    print ('val_split_bones_motion: {}'.format(val_split_bones_motion.shape))
    np.save(os.path.join(folder_out, 'val_data_joint.npy'), val_split_joints)
    np.save(os.path.join(folder_out, 'val_data_bone.npy'), val_split_bones)
    np.save(os.path.join(folder_out, 'val_data_joint_motion.npy'), val_split_joints_motion)
    np.save(os.path.join(folder_out, 'val_data_bone_motion.npy'), val_split_bones_motion)
    with open(os.path.join(folder_out, 'val_label.pkl'), 'wb') as f:
        pickle.dump((val_split_ok, val_labels_ok), f) 

    ## TEST SET
    if (proccess_test):
        test_split_joints = np.array(test_split_joints)
        test_split_bones = np.array(test_split_bones)
        test_split_joints_motion = np.array(test_split_joints_motion)
        test_split_bones_motion = np.array(test_split_bones_motion)
        print ('test_data_joint: {}'.format(test_split_joints.shape))
        print ('test_data_bone: {}'.format(test_split_bones.shape))
        print ('test_split_joints_motion: {}'.format(test_split_joints_motion.shape))
        print ('test_split_bones_motion: {}'.format(test_split_bones_motion.shape))
        np.save(os.path.join(folder_out, 'test_data_joint.npy'), test_split_joints)
        np.save(os.path.join(folder_out, 'test_data_bone.npy'), test_split_bones)
        np.save(os.path.join(folder_out, 'test_data_joint_motion.npy'), test_split_joints_motion)
        np.save(os.path.join(folder_out, 'test_data_bone_motion.npy'), test_split_bones_motion)
        with open(os.path.join(folder_out, 'test_label.pkl'), 'wb') as f:
            pickle.dump((test_split_ok, test_labels_ok), f)   


    ## TEST_MOVIL SET
    if (process_test_movil):
        test_movil_split_joints = np.array(test_movil_split_joints)
        test_movil_split_bones = np.array(test_movil_split_bones)
        test_movil_split_joints_motion = np.array(test_movil_split_joints_motion)
        test_movil_split_bones_motion = np.array(test_movil_split_bones_motion)
        print ('test_movil_data_joint: {}'.format(test_movil_split_joints.shape))
        print ('test_movil_data_bone: {}'.format(test_movil_split_bones.shape))
        print ('test_movil_split_joints_motion: {}'.format(test_movil_split_joints_motion.shape))
        print ('test_movil_split_bones_motion: {}'.format(test_movil_split_bones_motion.shape))
        np.save(os.path.join(folder_out, 'test_movil_data_joint.npy'), test_movil_split_joints)
        np.save(os.path.join(folder_out, 'test_movil_data_bone.npy'), test_movil_split_bones)
        np.save(os.path.join(folder_out, 'test_movil_data_joint_motion.npy'), test_movil_split_joints_motion)
        np.save(os.path.join(folder_out, 'test_movil_data_bone_motion.npy'), test_movil_split_bones_motion)
        with open(os.path.join(folder_out, 'test_movil_label.pkl'), 'wb') as f:
            pickle.dump((test_movil_split_ok, test_movil_labels_ok), f)   


    ## TEST_COLLABORATIONS SET
    if (process_test_collaborations):
        test_collaborations_split_joints = np.array(test_collaborations_split_joints)
        test_collaborations_split_bones = np.array(test_collaborations_split_bones)
        test_collaborations_split_joints_motion = np.array(test_collaborations_split_joints_motion)
        test_collaborations_split_bones_motion = np.array(test_collaborations_split_bones_motion)
        print ('test_collaborations_data_joint: {}'.format(test_collaborations_split_joints.shape))
        print ('test_collaborations_data_bone: {}'.format(test_collaborations_split_bones.shape))
        print ('test_collaborations_split_joints_motion: {}'.format(test_collaborations_split_joints_motion.shape))
        print ('test_collaborations_split_bones_motion: {}'.format(test_collaborations_split_bones_motion.shape))
        np.save(os.path.join(folder_out, 'test_collaborations_data_joint.npy'), test_collaborations_split_joints)
        np.save(os.path.join(folder_out, 'test_collaborations_data_bone.npy'), test_collaborations_split_bones)
        np.save(os.path.join(folder_out, 'test_collaborations_data_joint_motion.npy'), test_collaborations_split_joints_motion)
        np.save(os.path.join(folder_out, 'test_collaborations_data_bone_motion.npy'), test_collaborations_split_bones_motion)
        with open(os.path.join(folder_out, 'test_collaborations_label.pkl'), 'wb') as f:
            pickle.dump((test_collaborations_split_ok, test_collaborations_labels_ok), f)   

    ## ALL DATASET
    dataset_joints = np.array(dataset_joints)
    dataset_bones = np.array(dataset_bones)
    dataset_joints_motion = np.array(dataset_joints_motion)
    dataset_bones_motion = np.array(dataset_bones_motion)
    print ('dataset_joints: {}'.format(dataset_joints.shape))
    print ('dataset_bones: {}'.format(dataset_bones.shape))
    print ('dataset_joints_motion: {}'.format(dataset_joints_motion.shape))
    print ('dataset_bones_motion: {}'.format(dataset_bones_motion.shape))
    np.save(os.path.join(folder_out, 'dataset_joint.npy'), dataset_joints)
    np.save(os.path.join(folder_out, 'dataset_bone.npy'), dataset_bones)
    np.save(os.path.join(folder_out, 'dataset_joint_motion.npy'), dataset_joints_motion)
    np.save(os.path.join(folder_out, 'dataset_bone_motion.npy'), dataset_bones_motion)



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--extra', required=True, type=str)
    parser.add_argument('--folder_npy', required=True, type=str)
    parser.add_argument('--folder_labels', required=True, type=str)
    parser.add_argument('--folder_out', required=True, type=str)

    #arg = parser.parse_args([
    #    '--extra', "C3_xyz",
    #    '--folder_npy','/home/temporal2/mvazquez/bdd/AUTSL_mediapipe_Complex2/npy',
    #    '--folder_labels','/home/temporal2/mvazquez/bdd/AUTSL_mediapipe_Complex2/autsl_labels',
    #    '--folder_out','/home/temporal2/mvazquez/bdd/AUTSL_mediapipe_Complex2/data',
    #    ])
    arg = parser.parse_args()
    main(arg)
    
    #extra = ''
    #extra = 'C3_xyc'
    #extra = 'C3_xyz'
    #extra = 'C4_xyzc'

    """
    python generate_dataset.py \
        --extra C3_xyc \
        --folder_npy /home/temporal2/mvazquez/bdd/SignaMed/Signamed_mediapipe_Complex2/version1/npy \
        --folder_labels /home/temporal2/mvazquez/bdd/SignaMed/Signamed_mediapipe_Complex2/version1/labels \
        --folder_out /home/temporal2/mvazquez/bdd/SignaMed/Signamed_mediapipe_Complex2/version1/npy/data 

    python generate_dataset.py \
            --extra C3_xyz \
            --folder_npy /home/temporal2/mvazquez/bdd/SignaMed/Signamed_mediapipe_Complex2/version1/npy \
            --folder_labels /home/temporal2/mvazquez/bdd/SignaMed/Signamed_mediapipe_Complex2/version1/labels \
            --folder_out /home/temporal2/mvazquez/bdd/SignaMed/Signamed_mediapipe_Complex2/version1/npy/data 

    python generate_dataset.py \
            --extra C4_xyzc \
            --folder_npy /home/temporal2/mvazquez/bdd/SignaMed/Signamed_mediapipe_Complex2/version1/npy \
            --folder_labels /home/temporal2/mvazquez/bdd/SignaMed/Signamed_mediapipe_Complex2/version1/labels \
            --folder_out /home/temporal2/mvazquez/bdd/SignaMed/Signamed_mediapipe_Complex2/version1/npy/data 
    """