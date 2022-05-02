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
import time
from configs import Config
import mongoDBWrapper
import sys
import base64
sys.path.insert(0,'../compute_Service')
from preprocessing.src.gen_features import GenFeaturesMediapipe as Features
from preprocessing.src.gen_keypoints import GenKeypointsMediapipe as Keypoints

app = Flask(__name__)
ALLOWED_EXTENSIONS = {'mp4'}

signamedMongoDB = mongoDBWrapper.SignamedMongoDB()

config_file = 'conf/config_files.yaml'
configs = Config(config_file)

app.config['VIDEO_FOLDER'] = configs.video_folder
app.config['VIDEO_COLLABORATIONS_FOLDER'] = configs.video_collaborations_folder
app.config['MAX_CONTENT_LENGTH'] = configs.allowed_max_size
app.config['IP_SERVER_COMPUTE'] = configs.ip_server_compute

# db_register
with open(configs.dataset_json, 'r') as j:
    dataset_anns = json.loads(j.read())

gen_keypoints = Keypoints()
genFeatures = Features()

classes = list(dataset_anns.keys())
print (classes)
print (len(classes))
print (type(classes))


def generateFileNameTimestamp():
    return str(int(time.time()*1000000))
    
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

    if sign_id not in dataset_anns:
        return jsonify({"error": 'sign_id not registred'}),400 # Error 400: Bad Request
    if user_id == None:
        return jsonify({"error": 'Field user_id not found'}),400 # Error 400: Bad Request
    
    predictions = []
    predictions.append({
        "url_vid_sign": dataset_anns[sign_id]["url_vid_sign"],
        "url_vid_definition": dataset_anns[sign_id]["url_vid_definition"]
    })
    return jsonify({
        "user_id": user_id,
        "sign_id": sign_id,
        "urls": predictions,
        "definitions": dataset_anns[sign_id]["definition"],
        "signs": dataset_anns[sign_id]["name"]
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

    print ('get Features')
    keypoints = gen_keypoints.genKeypoints(os.path.join(app.config['VIDEO_FOLDER'], filename))
    print ('get Features')
    data_joints, data_bones = genFeatures.getFeatures(keypoints)

    url = 'http://{}/compute_only_msg3d'.format(app.config['IP_SERVER_COMPUTE'])
    resp = requests.post(url, data = {'data_joints': base64.b64encode(data_joints.tobytes()),'data_bones': base64.b64encode(data_bones.tobytes())})
    predictions = resp.json()['predictions'][0]
    probabilities = resp.json()['probabilities'][0]

    predictions_classes = []
    for p in predictions:
        predictions_classes.append(classes[p])

    signamedMongoDB.annotations_insert(identifier, user_id, str(float(identifier)/1000000), predictions_classes, probabilities)

    urls = []
    definitions = []
    signs = []
    for predict in predictions_classes:
        sign_id = predict
        urls.append({
            "url_vid_sign": dataset_anns[sign_id]["url_vid_sign"],
            "url_vid_definition": dataset_anns[sign_id]["url_vid_definition"]
        })
        definitions.append(dataset_anns[sign_id]["definition"])
        signs.append(dataset_anns[sign_id]["name"])
          
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

    
if __name__ == '__main__':

    # config_file = 'conf/config_files.yaml'
    # configs = Config(config_file)

    # os.environ["GOOGLE_APPLICATION_CREDENTIALS"]=configs.application_credentials
    # db_register = registerDB.RegisterAnnotations(configs.path_file_annotations)
    # with open(configs.dataset_json, 'r') as j:
    #     dataset_anns = json.loads(j.read())

    # app.config['VIDEO_FOLDER'] = configs.video_folder
    # #app.config['MAX_CONTENT_LENGTH'] = configs.allowed_max_size
    # app.config['IPS'] = configs.ip_servers

    # print (app.config['IPS'])
    # #app.run(host='0.0.0.0', port=56076, debug=True, ssl_context='adhoc')
    # context = ssl.SSLContext(ssl.PROTOCOL_TLSv1_2)
    # context.load_cert_chain("credentials/cert.pem", "credentials/key.pem")
    #app.run(host='0.0.0.0', port=56080, debug=True, use_reloader=False)

    app.run(host='0.0.0.0', port=56076, debug=True, use_reloader=False)
