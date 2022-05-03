import sys, os
import argparse
import numpy as np
import json
from configs import Config
import torch.nn.functional as F

sys.path.insert(0,'../../../compute_Service')
from preprocessing.src.gen_features import MediapipeOptions
from preprocessing.src.gen_features import GenFeaturesMediapipeC4 as Features
from preprocessing.src.gen_keypoints import GenKeypointsMediapipeC4 as Keypoints

sys.path.insert(0, '../../../compute_Service/msg3d')
from msg3d_processor2 import MSG3D_Processor

genFeatures = Features(MediapipeOptions.XYZ)

def main(parser):
    args = parser.parse_args()
    input = args.input
    config_model = args.config_model
    print (input)
    
    configs = Config(config_model)
    inputs = []
    models = []
    for config_msg3d_idx in configs.msg3d_configs_file:
        print ('config_msg3d_idx[0]: {}'.format(config_msg3d_idx[0]))
        print ('config_msg3d_idx[1]: {}'.format(config_msg3d_idx[1]))
        inputs.append(config_msg3d_idx[0])
        models.append(MSG3D_Processor(parser, config_msg3d_idx[1]))
        print ('input: {}'.format(config_msg3d_idx[0]))
        print ('model: {}'.format(config_msg3d_idx[1]))
        
    with open(input, "r") as f:
        data = json.loads(f.read())
    keypoints = np.asarray(data, dtype=np.float64)
    data_joints, data_bones, data_motion_joints, data_motion_bones = genFeatures.getFeatures(keypoints)
    inputs_features = {
        'data_joints' : data_joints,
        'data_bones' : data_bones,
        'data_motion_joints' : data_motion_joints,
        'data_motion_bones' : data_motion_bones
    }
    
    output_ensemble = []
    for idx in range(len(inputs)):
        print ('configs.channels: ',configs.channels)
        print ('configs.frames: ',configs.frames)
        print ('configs.num_kps: ',configs.num_kps)
        print ('get inference')
        index, prob, output, output_normal, output_flipped = models[idx].inference(inputs_features[inputs[idx]])
        print ('MS-G3D {} - prob: {} & index: {}'.format(inputs[idx],prob,index))

        if idx == 0:
            output_ensemble = output_normal
        else:
            output_ensemble = (output_ensemble + output_normal) / 2

    output_ensemble = F.softmax(output_ensemble[0], dim=0)
    output_ensemble = output_ensemble.reshape(1, len(output_ensemble))
    prob, index = output_ensemble.topk(3)

    print ('MS-G3D || Result :  prob: {} & index: {}'.format(index.tolist(),prob.tolist()))
        
        

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Generate Result Output')
    parser.add_argument('--input', required=True, default='', type=str)
    parser.add_argument('--config_model', required=False, default='config/app_compute_config.yaml', type=str)
    
    # Herencia "msg3d processor".
    parser.add_argument(
        '--model',
        default=None,
        help='the model will be used')
    parser.add_argument(
        '--model-args',
        type=dict,
        default=dict(),
        help='the arguments of model')
    parser.add_argument(
        '--weights',
        default=None,
        help='the weights for network initialization')
    parser.add_argument(
        '--half',
        action='store_true',
        help='Use half-precision (FP16) training')
    parser.add_argument(
        '--device',
        type=int,
        default=0,
        nargs='+',
        help='the indexes of GPUs for training or testing')
    
    main(parser)
    
    
# python generate_prediction.py --input ../data/kps53/temp_afectar.json
