#!/usr/bin/env python
import os
import yaml
import pickle
import argparse
from collections import OrderedDict, defaultdict
import torch
from torchsummary import summary
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

# def get_parser():
#     # parameter priority: command line > config file > default
#     parser = argparse.ArgumentParser(description='MS-G3D')
#     parser.add_argument('--input', required=False, type=str)
#     parser.add_argument('--output', required=False, type=str)
#     parser.add_argument('--model_config', required=False, type=str)
#     parser.add_argument('--type_features', nargs="+", required=False)
#     parser.add_argument('--folder_models', required=False, type=str)
    
#     parser.add_argument(
#         '--config',
#         default='/home/bdd/LSE_Lex40_uvigo/dataconfig/nturgbd-cross-view/test_bone.yaml',
#         help='path to the configuration file')
#     parser.add_argument(
#         '--model',
#         default=None,
#         help='the model will be used')
#     parser.add_argument(
#         '--model-args',
#         type=dict,
#         default=dict(),
#         help='the arguments of model')
#     parser.add_argument(
#         '--weights',
#         default=None,
#         help='the weights for network initialization')
#     parser.add_argument(
#         '--half',
#         action='store_true',
#         help='Use half-precision (FP16) training')
#     parser.add_argument(
#         '--device',
#         type=int,
#         default=0,
#         nargs='+',
#         help='the indexes of GPUs for training or testing')
#     return parser

def import_class(name):
    components = name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod

def flip(data_numpy):
    
    flipped_data = np.copy(data_numpy)
    flipped_data[0,:,:,:] *= -1
    
    return flipped_data

class MSG3D_Processor():
    """Processor for Skeleton-based Action Recgnition"""

    def __init__(self, parser, config):
        
        # parser = get_parser()
        p = parser.parse_args()
        with open(config, 'r') as f:
            default_arg = yaml.load(f)
            key = vars(p).keys()
        parser.set_defaults(**default_arg)
        self.arg = parser.parse_args()
        self.load_model()
        self.model.eval()

        if type(self.arg.device) is list:
            if len(self.arg.device) > 1:
                self.print_log(f'{len(self.arg.device)} GPUs available, using DataParallel')
                self.model = nn.DataParallel(
                    self.model,
                    device_ids=self.arg.device,
                    output_device=self.output_device
                )

    def load_model(self):
        print ('loading model...')
        output_device = self.arg.device[0] if type(
            self.arg.device) is list else self.arg.device
        self.output_device = output_device
        Model = import_class(self.arg.model)

        self.model = Model(**self.arg.model_args).cuda(output_device)
        self.loss = nn.CrossEntropyLoss().cuda(output_device)
        
        if self.arg.weights:
            try:
                self.global_step = int(arg.weights[:-3].split('-')[-1])
            except:
                print('Cannot parse global_step from model weights filename')
                self.global_step = 0

            print('Loading weights from: {}'.format(self.arg.weights))
            if '.pkl' in self.arg.weights:
                with open(self.arg.weights, 'r') as f:
                    weights = pickle.load(f)
            else:
                weights = torch.load(self.arg.weights)

            weights = OrderedDict(
                [[k.split('module.')[-1],
                  v.cuda(output_device)] for k, v in weights.items()])

            try:
                self.model.load_state_dict(weights)
            except:
                state = self.model.state_dict()
                diff = list(set(state.keys()).difference(set(weights.keys())))
                print('Can not find these weights:')
                for d in diff:
                    self.print_log('  ' + d)
                state.update(weights)
                self.model.load_state_dict(state)
                
        for param in self.model.parameters():
                param.requires_grad = False

    def inference(self, data):
        data_flip = np.copy(data)
        data_flip = flip(data_flip)

        data = torch.from_numpy(np.array([data]))
        data_flip = torch.from_numpy(np.array([data_flip]))
        
        self.model = self.model.cuda(self.output_device)
        data = data.float().cuda(self.output_device)
        data_flip = data_flip.float().cuda(self.output_device)
        
        # print ('data_shape: {}', data.shape)
        # print ('data_flipped_shape: {}', data_flip.shape)
        #summary(self.model, (3, 157, 51, 1))

        output = self.model(data)
        
        # Disabled use of flipped and combinate them. Use only normal data.
        #output_flipped = output
        #output_tta = output
        output_flipped = self.model(data_flip)
        output_tta = (output +  output_flipped) / 2
                            
        output_softmax = F.softmax(output_tta[0], dim=0)
        output_softmax = output_softmax.reshape(1, len(output_softmax))
        
        prob, indices = output_softmax.topk(3)
            
        return indices.cpu().numpy()[0], prob.cpu().numpy()[0], output_tta, output, output_flipped
    
    def inference_softmax(self, data):
        data_flip = np.copy(data)
        data_flip = flip(data_flip)

        data = torch.from_numpy(np.array([data]))
        data_flip = torch.from_numpy(np.array([data_flip]))
        
        self.model = self.model.cuda(self.output_device)
        data = data.float().cuda(self.output_device)
        data_flip = data_flip.float().cuda(self.output_device)
        
        # print ('data_shape: {}', data.shape)
        # print ('data_flipped_shape: {}', data_flip.shape)
        #summary(self.model, (3, 157, 51, 1))

        output = self.model(data)
        
        # Disabled use of flipped and combinate them. Use only normal data.
        #output_flipped = output
        #output_tta = output
        output_flipped = self.model(data_flip)
        output_tta = (output +  output_flipped) / 2
                            
        output_softmax = F.softmax(output_tta[0], dim=0)
        output_softmax = output_softmax.reshape(1, len(output_softmax))
        
        prob, indices = output_softmax.topk(82)
            
        return indices.cpu().numpy()[0], prob.cpu().numpy()[0], output_softmax

    def inference_softmax_with_batch_size(self, data):
        if data.ndim == 4:
            data = np.array([data])
        
        data_flip = np.copy(data)
        data_flip = flip(data_flip)

        data = torch.from_numpy(np.array(data))
        data_flip = torch.from_numpy(np.array(data_flip))
        
        self.model = self.model.cuda(self.output_device)
        data = data.float().cuda(self.output_device)
        data_flip = data_flip.float().cuda(self.output_device)
        output = self.model(data)
                
        output_flipped = self.model(data_flip)
        output_tta = (output +  output_flipped) / 2

        arr_output_softmax = []
        arr_prob = []
        arr_indices = []
        for out_idx in range(data.shape[0]):
            output_softmax = F.softmax(output_tta[out_idx], dim=0)
            output_softmax = output_softmax.reshape(1, len(output_softmax))
            prob, indices = output_softmax.topk(82)
            
            arr_output_softmax.append(output_softmax)
            arr_indices.append(indices.cpu().numpy()[0])
            arr_prob.append(prob.cpu().numpy()[0])
            
        return arr_indices, arr_prob, arr_output_softmax
    

def main():
    
    config = '/home/temporal2/mvazquez/CSLR_workspace/SignSpotting/src/config/model/signamed20fps/joints_C3_xyz.yaml'
    processor = MSG3D_Processor(config)
    folder_features_joints_xyz = '/home/temporal2/mvazquez/CSLR_workspace/SignSpotting/experiments/EXPERIMENTO_17MARZO/B1_generate_windows/joints_C3_xyz'
    list_files = os.listdir(folder_features_joints_xyz)
    
    #for vid in list_files:
    fidx = list_files[0]
    fidx_path = os.path.join(folder_features_joints_xyz, fidx)
    
    print ('path: {}'.format(fidx))
    
    data_in = np.load(fidx_path)
    print ('data_in shape: {}'.format(data_in.shape))
    
    BATCH_SIZE = 5
    indices, prob, out = processor.inference_softmax_with_batch_size(data_in[0:BATCH_SIZE])
    
    print ('prob: {}'.format(prob))
    print ('index: {}'.format(indices))
    
    print ('out: {}'.format(out))


if __name__ == '__main__':
    main()