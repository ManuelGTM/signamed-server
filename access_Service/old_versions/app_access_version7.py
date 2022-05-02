import os
import numpy as np
from google.cloud import storage
from flask import Flask, jsonify, request
from werkzeug.utils import secure_filename
import datetime
import json
import requests
import time
import ssl
import yaml
import time
import mongoDBWrapper
import sys
import base64
sys.path.insert(0,'../compute_Service')
from preprocessing.src.gen_features import MediapipeOptions
from preprocessing.src.gen_features import GenFeaturesMediapipeC4 as Features
from preprocessing.src.gen_keypoints import GenKeypointsMediapipeC4 as Keypoints

app = Flask(__name__)

signamedMongoDB = mongoDBWrapper.SignamedMongoDB()

config_path = 'conf/config_files.yaml'
with open(config_path, "r") as ymlfile:
    cfg = yaml.load(ymlfile)

app.config['VIDEO_FOLDER'] = cfg['VIDEO_FOLDER'] 
app.config['VIDEO_COLLABORATIONS_FOLDER'] = cfg['VIDEO_COLAB_FOLDER']
app.config['MAX_CONTENT_LENGTH'] = cfg['ALLOWED_MAX_SIZE']
app.config['IP_SERVER_COMPUTE'] = cfg['IP_SERVER_COMPUTE']
ALLOWED_EXTENSIONS = cfg['ALLOWED_EXTENSIONS']

# db_register
with open(cfg['DATASET_JSON'], 'r') as j:
    dataset_anns = json.loads(j.read())

gen_keypoints = Keypoints()
genFeatures = Features(MediapipeOptions.XYZ)

classes = []
for sign in dataset_anns:
    classes.append(sign['_id'])

print (classes)
print ('Number of classes: {}'.format(len(classes)))


def generateFileNameTimestamp():
    return str(int(time.time()*1000000))

def search_sign_id(sign_id):
    return [element for element in dataset_anns if element['_id'] == sign_id]
    
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/hola')
def hola():
    msg = 'hola: {}'.format(generateFileNameTimestamp())

    return msg

@app.route('/sendText', methods=["GET", "POST"])
def getPrediction_sendText():
    user_id = request.form.get('user_id')
    sign_id = request.form.get('sign_id')
    print ('user_id: {}'.format(user_id))
    print ('sign_id: {}'.format(sign_id))
    if sign_id == None:
        return jsonify({"error": 'Field sign_id not found'}),400 # Error 400: Bad Request
    if user_id == None:
        return jsonify({"error": 'Field user_id not found'}),400 # Error 400: Bad Request

    info = search_sign_id(sign_id)
    if len(info)==0:
        return jsonify({"error": 'sign_id not registred'}),400 # Error 400: Bad Request
    
   
    predictions = []
    predictions.append({
        "url_vid_sign": info[0]["url_vid_sign"],
        "url_vid_definition": info[0]["url_vid_definition"]
    })
    return jsonify({
        "user_id": user_id,
        "sign_id": sign_id,
        "urls": predictions,
        "definitions": info[0]["definition"],
        "signs": info[0]["name"]
        })
    
@app.route('/sendVideo', methods=["POST"])
def sendVideo():
    user_id = request.form.get('user_id')
    file = request.files['file']
    
    if file == None:
        return jsonify({"error": 'Field file not found'}),400 # Error 400: Bad Request
    
    print ('file.filename: {}'.format(file.filename))
    if file.filename == '':
        return jsonify({"error": 'Field file empty'}),400 # Error 400: Bad Request
    if file and allowed_file(file.filename):
        identifier = generateFileNameTimestamp()
        filename = identifier +'.mp4'
        file.save(os.path.join(app.config['VIDEO_FOLDER'], filename))
        print ('save: {}'.format(os.path.join(app.config['VIDEO_FOLDER'], filename)))
    else:
        return jsonify({"error": 'Field file format not allowed'}),400 # Error 400: Bad Request

    print ('get Keypoints')
    keypoints = gen_keypoints.genKeypoints(os.path.join(app.config['VIDEO_FOLDER'], filename))
    print ('get Features')
    data_joints, data_bones, data_motion_joints, data_motion_bones = genFeatures.getFeatures(keypoints)

    print ('data_joints shape: {}'.format(data_joints.shape))
    print ('data_bones shape: {}'.format(data_bones.shape))
    print ('data_motion_joints shape: {}'.format(data_motion_joints.shape))
    print ('data_motion_bones shape: {}'.format(data_joints.shape))

    url = 'http://{}/compute_only_msg3d'.format(app.config['IP_SERVER_COMPUTE'])
    resp = requests.post(url, data = {'data_joint': base64.b64encode(data_joints.tobytes()),'data_bone': base64.b64encode(data_bones.tobytes()),'data_joint_motion': base64.b64encode(data_motion_joints.tobytes()),'data_bone_motion': base64.b64encode(data_motion_bones.tobytes())})
    predictions = resp.json()['predictions'][0]
    probabilities = resp.json()['probabilities'][0]

    predictions_classes = []
    for p in predictions:
        predictions_classes.append(classes[p])

    signamedMongoDB.annotations_insert(identifier, user_id, str(float(identifier)/1000000), predictions_classes, probabilities)

    urls = []
    definitions = []
    signs = []
    for p_idx in predictions:
        urls.append({
            "url_vid_sign": dataset_anns[p_idx]["url_vid_sign"],
            "url_vid_definition": dataset_anns[p_idx]["url_vid_definition"]
        })
        definitions.append(dataset_anns[p_idx]["definition"])
        signs.append(dataset_anns[p_idx]["name"])
          
    return jsonify({
        "user_id": user_id,
        "video_id": identifier,
        "predictions": predictions_classes,
        "probabilities": probabilities,
        "urls": urls,
        "definitions": definitions,
        "signs": signs
        })
    
@app.route('/sendCollaboration', methods=["POST"])
def sendCollaboration():
    user_id = request.form.get('user_id')
    sign_id = request.form.get('sign_id')
    file = request.files['file']
    
    if file == None:
        return jsonify({"error": 'Field file not found'}),400 # Error 400: Bad Request
    
    print ('file.filename: {}'.format(file.filename))
    if file.filename == '':
        return jsonify({"error": 'Field file empty'}),400 # Error 400: Bad Request
    if file and allowed_file(file.filename):
        identifier = generateFileNameTimestamp()
        filename = identifier +'.mp4'
        file.save(os.path.join(app.config['VIDEO_COLLABORATIONS_FOLDER'], filename))
        print ('save: {}'.format(os.path.join(app.config['VIDEO_COLLABORATIONS_FOLDER'], filename)))
    else:
        return jsonify({"error": 'Field file format not allowed'}),400 # Error 400: Bad Request

    signamedMongoDB.collaborations_insert(identifier, user_id, str(float(identifier)/1000000), sign_id)

    return jsonify({
        "user_id": user_id,
        "video_id": identifier
        })


@app.route('/feedback', methods=["GET", "POST"])
def processFeedback():

    user_id = request.form.get('user_id')
    video_id = request.form.get('video_id')
    feedback_prediction = request.form.get('feedback_prediction')


    if signamedMongoDB.annotations_update_feedback(video_id, user_id, feedback_prediction):
        return jsonify({
            "user_id": user_id,
            "video_id": video_id,
            "prediction": feedback_prediction,
            })
    else:
        return jsonify({"error": 'Entry not found'}),400 # Error 400: Bad Request


@app.route('/sendMediapipe', methods=["GET", "POST"])
def sendMediapipe():
    print ('sendMediapipe')
    user_id = request.form.get('user_id')
    file = request.files['mediapipe']
    file.save('temp.json')
    with open('temp.json', "r") as f:
        data = json.loads(f.read())
    keypoints = np.asarray(data, dtype=np.float64)
    identifier = generateFileNameTimestamp()

    print ('get Features')
    data_joints, data_bones, data_motion_joints, data_motion_bones = genFeatures.getFeatures(keypoints)

    print ('data_joints shape: {}'.format(data_joints.shape))
    print ('data_bones shape: {}'.format(data_bones.shape))
    print ('data_motion_joints shape: {}'.format(data_motion_joints.shape))
    print ('data_motion_bones shape: {}'.format(data_joints.shape))
    
    url = 'http://{}/compute_only_msg3d'.format(app.config['IP_SERVER_COMPUTE'])
    resp = requests.post(url, data = {'data_joint': base64.b64encode(data_joints.tobytes()),'data_bone': base64.b64encode(data_bones.tobytes()),'data_joint_motion': base64.b64encode(data_motion_joints.tobytes()),'data_bone_motion': base64.b64encode(data_motion_bones.tobytes())})
    predictions = resp.json()['predictions'][0]
    probabilities = resp.json()['probabilities'][0]

    print (predictions)
    print (probabilities)

    predictions_classes = []
    for p in predictions:
        predictions_classes.append(classes[p])

    #signamedMongoDB.annotations_insert(identifier, user_id, str(float(identifier)/1000000), predictions_classes, probabilities)

    urls = []
    definitions = []
    signs = []
    for p_idx in predictions:
        urls.append({
            "url_vid_sign": dataset_anns[p_idx]["url_vid_sign"],
            "url_vid_definition": dataset_anns[p_idx]["url_vid_definition"]
        })
        definitions.append(dataset_anns[p_idx]["definition"])
        signs.append(dataset_anns[p_idx]["name"])
          
    return jsonify({
        "user_id": user_id,
        "video_id": identifier,
        "predictions": predictions_classes,
        "probabilities": probabilities,
        "urls": urls,
        "definitions": definitions,
        "signs": signs
        })


    #mediapipe = request.files['mediapipe']
    # mediapipe = request.files['mediapipe']
    #msg = 'hola: {}'.format(generateFileNameTimestamp())
    #return msg
    
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=56076, debug=True, use_reloader=False)

