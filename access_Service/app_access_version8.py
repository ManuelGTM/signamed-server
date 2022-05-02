import base64
import sys
import mongoDBWrapper
import yaml
import ssl
import time
import requests
import json
import datetime
import random
from werkzeug.utils import secure_filename
from flask import Flask, jsonify, request, send_file, send_from_directory
from google.cloud import storage
import numpy as np
import os
sys.path.insert(0, '../compute_Service')
from preprocessing.src.gen_keypoints import GenKeypointsMediapipeC4 as Keypoints
from preprocessing.src.gen_features import GenFeaturesMediapipeC4 as Features
from preprocessing.src.gen_features import MediapipeOptions


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

print(classes)
print('Number of classes: {}'.format(len(classes)))


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
    print('user_id: {}'.format(user_id))
    print('sign_id: {}'.format(sign_id))
    if sign_id == None:
        # Error 400: Bad Request
        return jsonify({"error": 'Field sign_id not found'}), 400
    if user_id == None:
        # Error 400: Bad Request
        return jsonify({"error": 'Field user_id not found'}), 400

    info = search_sign_id(sign_id)
    if len(info) == 0:
        # Error 400: Bad Request
        return jsonify({"error": 'sign_id not registred'}), 400

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
        # Error 400: Bad Request
        return jsonify({"error": 'Field file not found'}), 400

    print('file.filename: {}'.format(file.filename))
    if file.filename == '':
        # Error 400: Bad Request
        return jsonify({"error": 'Field file empty'}), 400
    if file and allowed_file(file.filename):
        identifier = generateFileNameTimestamp()
        filename = identifier + '.mp4'
        file.save(os.path.join(app.config['VIDEO_FOLDER'], filename))
        print('save: {}'.format(os.path.join(
            app.config['VIDEO_FOLDER'], filename)))
    else:
        # Error 400: Bad Request
        return jsonify({"error": 'Field file format not allowed'}), 400

    print('get Keypoints')
    keypoints = gen_keypoints.genKeypoints(
        os.path.join(app.config['VIDEO_FOLDER'], filename))
    print('get Features')
    data_joints, data_bones, data_motion_joints, data_motion_bones = genFeatures.getFeatures(
        keypoints)

    print('data_joints shape: {}'.format(data_joints.shape))
    print('data_bones shape: {}'.format(data_bones.shape))
    print('data_motion_joints shape: {}'.format(data_motion_joints.shape))
    print('data_motion_bones shape: {}'.format(data_joints.shape))

    url = 'http://{}/compute_only_msg3d'.format(
        app.config['IP_SERVER_COMPUTE'])
    resp = requests.post(url, data={'data_joint': base64.b64encode(data_joints.tobytes()), 'data_bone': base64.b64encode(data_bones.tobytes(
    )), 'data_joint_motion': base64.b64encode(data_motion_joints.tobytes()), 'data_bone_motion': base64.b64encode(data_motion_bones.tobytes())})
    predictions = resp.json()['predictions'][0]
    probabilities = resp.json()['probabilities'][0]

    predictions_classes = []
    for p in predictions:
        predictions_classes.append(classes[p])

    signamedMongoDB.insert_inference(identifier, user_id, str(
        float(identifier)/1000000), predictions_classes, probabilities)

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
        # Error 400: Bad Request
        return jsonify({"error": 'Field file not found'}), 400

    print('file.filename: {}'.format(file.filename))
    if file.filename == '':
        # Error 400: Bad Request
        return jsonify({"error": 'Field file empty'}), 400
    if file and allowed_file(file.filename):
        identifier = generateFileNameTimestamp()
        filename = identifier + '.mp4'
        file.save(os.path.join(
            app.config['VIDEO_COLLABORATIONS_FOLDER'], filename))
        print('save: {}'.format(os.path.join(
            app.config['VIDEO_COLLABORATIONS_FOLDER'], filename)))
    else:
        # Error 400: Bad Request
        return jsonify({"error": 'Field file format not allowed'}), 400

    signamedMongoDB.insert_collaborations(
        identifier, user_id, str(float(identifier)/1000000), sign_id)

    return jsonify({
        "user_id": user_id,
        "video_id": identifier
    })


@app.route('/feedback', methods=["GET", "POST"])
def processFeedback():

    user_id = request.form.get('user_id')
    video_id = request.form.get('video_id')
    feedback_prediction = request.form.get('feedback_prediction')

    if signamedMongoDB.update_inference_feedback(video_id, user_id, feedback_prediction):
        return jsonify({
            "user_id": user_id,
            "video_id": video_id,
            "prediction": feedback_prediction,
        })
    else:
        # Error 400: Bad Request
        return jsonify({"error": 'Entry not found'}), 400


@app.route('/sendMediapipe', methods=["GET", "POST"])
def sendMediapipe():
    print('sendMediapipe')
    user_id = request.form.get('user_id')
    file = request.files['mediapipe']
    file.save('temp.json')
    with open('temp.json', "r") as f:
        data = json.loads(f.read())
    keypoints = np.asarray(data, dtype=np.float64)
    identifier = generateFileNameTimestamp()

    print('get Features')
    data_joints, data_bones, data_motion_joints, data_motion_bones = genFeatures.getFeatures(
        keypoints)

    print('data_joints shape: {}'.format(data_joints.shape))
    print('data_bones shape: {}'.format(data_bones.shape))
    print('data_motion_joints shape: {}'.format(data_motion_joints.shape))
    print('data_motion_bones shape: {}'.format(data_joints.shape))

    url = 'http://{}/compute_only_msg3d'.format(
        app.config['IP_SERVER_COMPUTE'])
    resp = requests.post(url, data={'data_joint': base64.b64encode(data_joints.tobytes()), 'data_bone': base64.b64encode(data_bones.tobytes(
    )), 'data_joint_motion': base64.b64encode(data_motion_joints.tobytes()), 'data_bone_motion': base64.b64encode(data_motion_bones.tobytes())})
    predictions = resp.json()['predictions'][0]
    probabilities = resp.json()['probabilities'][0]

    print(predictions)
    print(probabilities)

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


@app.route('/getVideoDatasetBuilder', methods=["GET", "POST"])
def requestVideoDatasetBuilder():

    print('getVideoDatasetBuilder')

    user_id = request.form.get('user_id')
    session = request.form.get('session')
    filename = request.form.get('filename')
    
    print ('user_id: '+user_id)
    print ('session: '+session)
    print ('filename: '+filename)
    
    dataset_builder_session = signamedMongoDB.get_DasetBuilder_Session(session)
    folder_path = dataset_builder_session['path']
    
    print ('folder_path: '+folder_path)
    print ('filename: '+filename)
    
    # filename_random = random.choice(os.listdir(folder_path))

    # filename = '1621388235484450.mp4'
    # return send_file(safe_path, as_attachment=True)
    return send_from_directory(folder_path, filename, mimetype='video/mp4')

@app.route('/sendDatasetBuilder', methods=["GET", "POST"])
def sendDatasetBuilder():
    
    print('sendDatasetBuilder')

    user_id = request.form.get('user_id')
    session = request.form.get('session')
    filename = request.form.get('filename')
    class_id = request.form.get('class_id')
    annotation_code_id = request.form.get('annotation_code_id')
    comment = request.form.get('comment')
    counter_repeats_used = request.form.get('counter_repeats_used')
    history_log = request.form.get('history_log')
    
    print ('user_id: '+user_id)
    print ('session: '+session)
    print ('class_id: '+class_id)
    print ('annotation_code_id: '+annotation_code_id)
    
    try:
        print ('comment: '+comment)
    except:
        comment = ''
        print ('comment: comment invalid, it was deleted')
        
    print ('counter_repeats_used: '+counter_repeats_used)
    print ('history_log: '+history_log)
    
    list_history_log = []
    print (type(list_history_log))
    if (len(history_log) > 0):
        list_history_log = history_log.split(',')
        
    print ('list_history_log: ')
    print (list_history_log)
       
    # INSERT ANNOTATION
    print ('COMPROBACION')
    
    class_id = int(class_id)
    annotation_code_id = int(annotation_code_id)
    
    if (class_id == -1 and annotation_code_id == -1):
        print ('JUMP : No actions')
    else:
        print ('insert new')
        counter_repeats_used = int(counter_repeats_used)
        signamedMongoDB.insert_DatasetBuilder_Annotation(user_id, session, filename, class_id, annotation_code_id, comment, counter_repeats_used);
        signamedMongoDB.update_DatasetBuilder_Annotation_Counters(session);
        
        
    # METODO PARA DETERMINAR CUAL ES EL FICHERO QUE ENVIAMOS.
    # ALEATORIO.
    dataset_builder_session = signamedMongoDB.get_DasetBuilder_Session(session)
    data_idx = signamedMongoDB.get_DatasetBuilder_next_Video_Improved_Method(dataset_builder_session, user_id,list_history_log)
    
    
    return jsonify({
        "filename": data_idx['filename'],
        "class_ref" : data_idx['class_ref'],
        "info": data_idx['info'],
        "classes": dataset_builder_session['classes'],
        "annotation_codes": dataset_builder_session['annotation_codes'],
        "session_url" : dataset_builder_session['session_url']
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=56076, debug=True, use_reloader=False)
