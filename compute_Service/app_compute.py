import os, sys
import numpy as np
from flask import Flask, jsonify, request
import torch.nn.functional as F
import base64
from msg3d.msg3d_processor import MSG3D_Processor
from configs import Config
sys.path.insert(0, './msg3d')


config_file = 'config/app_compute_config.yaml'
configs = Config(config_file)
ip_compute_msg3d = configs.ip_server_compute.split(':')[0]
port_compute_msg3d = configs.ip_server_compute.split(':')[1]

# SERVICIO REST.
app = Flask(__name__)

inputs = []
models = []
for config_msg3d_idx in configs.msg3d_configs_file:
    print ('config_msg3d_idx[0]: {}'.format(config_msg3d_idx[0]))
    print ('config_msg3d_idx[1]: {}'.format(config_msg3d_idx[1]))
    inputs.append(config_msg3d_idx[0])
    models.append(MSG3D_Processor(config_msg3d_idx[1]))
    print ('input: {}'.format(config_msg3d_idx[0]))
    print ('model: {}'.format(config_msg3d_idx[1]))

# config_msg3d_joints = os.path.join(configs.msg3d_configs_file, 'test_joint_tta.yaml')
# config_msg3d_bones = os.path.join(configs.msg3d_configs_file, 'test_bone_tta.yaml')
# msg3d_joints = MSG3D_Processor(config_msg3d_joints)
# msg3d_bones = MSG3D_Processor(config_msg3d_bones)

@app.route('/available')
def available():
    return True 

# @app.route('/compute_only_msg3d', methods=["POST"])
# def compute2():
#     data_joints = np.frombuffer(base64.b64decode(request.form.get('data_joints'))).reshape(configs.channels, configs.frames, configs.num_kps, 1)
#     data_bones = np.frombuffer(base64.b64decode(request.form.get('data_bones'))).reshape(configs.channels, configs.frames, configs.num_kps, 1)
        
#     print ('get inference')
#     index, prob, output_joints, output_joints_normal , output_joints_flipped = msg3d_joints.inference(data_joints)
#     print ('MS-G3D JOINTS - prob: {} & index: {}'.format(prob,index))
    
#     index, prob, output_bones, output_bones_normal , output_bones_flipped = msg3d_bones.inference(data_bones)
#     print ('MS-G3D BONES - prob: {} & index: {}'.format(prob,index))
    
#     output_ensemble = (output_joints +  output_bones) / 2
#     output_ensemble = F.softmax(output_ensemble[0], dim=0)
#     output_ensemble = output_ensemble.reshape(1, len(output_ensemble))
#     prob, index = output_ensemble.topk(3)
             
#     return jsonify({
#         "predictions": index.tolist(),
#         "probabilities": prob.tolist()
#         })

@app.route('/compute_only_msg3d', methods=["POST"])
def compute_only_msg3d_v2():
    print ('compute_only_msg3d')
    output_ensemble = []
    for idx in range(len(inputs)):
        print ('configs.channels: ',configs.channels)
        print ('configs.frames: ',configs.frames)
        print ('configs.num_kps: ',configs.num_kps)

        data = np.frombuffer(base64.b64decode(request.form.get(inputs[idx]))).reshape(configs.channels, configs.frames, configs.num_kps, 1)
        print ('get inference')
        index, prob, output, output_normal, output_flipped = models[idx].inference(data)
        print ('MS-G3D {} - prob: {} & index: {}'.format(inputs[idx],prob,index))

        if idx == 0:
            output_ensemble = output_normal
        else:
            output_ensemble = (output_ensemble + output_normal) / 2

    output_ensemble = F.softmax(output_ensemble[0], dim=0)
    output_ensemble = output_ensemble.reshape(1, len(output_ensemble))
    prob, index = output_ensemble.topk(3)

    print ('MS-G3D || Result :  prob: {} & index: {}'.format(index.tolist(),prob.tolist()))
             
    return jsonify({
        "predictions": index.tolist(),
        "probabilities": prob.tolist()
        })

    
if __name__ == '__main__':
    app.run(host=ip_compute_msg3d, port=port_compute_msg3d, debug=True, use_reloader=False)
    #print (configs.msg3d_configs_file)
    #for model in configs.msg3d_configs_file:
    #    print (model)
    
