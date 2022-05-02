import os
import numpy as np
from google.cloud import storage
from flask import Flask, jsonify, request
from werkzeug.utils import secure_filename
from configs import Config
import datetime
import json
import requests
import time
import registerDB
import ssl

ALLOWED_EXTENSIONS = {'mp4'}

# GOOGLE CLOUD STORAGE.
def generate_download_signed_url_v4(sign_id, configs, dataset_anns):
    
    storage_client = storage.Client()
    bucket = storage_client.bucket(configs.bucket_sign)

    url_sign = None
    url_definition = None

    try:
        if (dataset_anns[sign_id]["vid_sign"]!=""):
            blob = bucket.blob(dataset_anns[sign_id]["vid_sign"])
            url_sign = blob.generate_signed_url(
                version="v4",
                expiration=datetime.timedelta(minutes=15),
                # expiration=datetime.timedelta(days=6),
                method="GET",
            )
                
            bucket = storage_client.bucket(configs.bucket_definition)
                
            blob = bucket.blob(dataset_anns[sign_id]["vid_definition"])
            url_definition = blob.generate_signed_url(
                version="v4",
                # This URL is valid for 15 minutes
                expiration=datetime.timedelta(minutes=15),
                # expiration=datetime.timedelta(days=6),
                method="GET",
            )
    except:
        pass

    return url_sign, url_definition

def generateFileNameTimestamp():
    return str(int(time.time()*1000000))
    
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# SERVICIO REST.
app = Flask(__name__)

@app.route('/sendText')
def getPrediction_sendText():

    user_id = request.form.get('user_id')
    sign_id = request.form.get('sign_id')
    print ('user_id: {}'.format(user_id))
    print ('sign_id: {}'.format(sign_id))
    
    if sign_id == None:
        return jsonify({"error": 'Field sign_id not found'}),400 # Error 400: Bad Request
    
    sign_id = str(sign_id).zfill(5)
    if sign_id not in dataset_anns:
        return jsonify({"error": 'sign_id not registred'}),400 # Error 400: Bad Request
    if user_id == None:
        return jsonify({"error": 'Field user_id not found'}),400 # Error 400: Bad Request
    
    url_vid_sign, url_vid_def = generate_download_signed_url_v4(sign_id, configs, dataset_anns)
    
    predictions = []
    predictions.append({
        "url_vid_sign": url_vid_sign,
        "url_vid_definition": url_vid_def
    })
    return jsonify({
        "user_id": user_id,
        "sign_id": sign_id,
        "predictions": predictions
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

    url = 'https://{}/compute'.format(app.config['IPS'][0])
    resp = requests.post(url, files={'file': open(os.path.join(app.config['VIDEO_FOLDER'], filename))},verify=False)
    predictions = resp.json()['predictions']
    probabilities = resp.json()['probabilities']

    anns = registerDB.STRUCTURE_ANNS
    anns['identifier'] = identifier
    anns['user_id'] = user_id
    anns['timestamp'] = float(identifier)/1000000
    anns['predictions'] = predictions
    anns['probabilities'] = probabilities
    anns['feedback'] = -1
    db_register.push(anns)

    urls = []
    for predict in predictions:
        sign_id = str(predict).zfill(5)
        url_vid_sign, url_vid_def = generate_download_signed_url_v4(sign_id,configs, dataset_anns)
        urls.append({
            "url_vid_sign": url_vid_sign,
            "url_vid_definition": url_vid_def
        })
          
    return jsonify({
        "user_id": user_id,
        "video_id": identifier,
        "predictions": predictions,
        "probabilities": probabilities,
        "urls": urls
        })
    
@app.route('/feedback', methods=["GET"])
def processFeedback():
    user_id = request.form.get('user_id').encode('ascii','ignore')
    video_id = request.form.get('video_id').encode('ascii','ignore')
    feedback_prediction = request.form.get('feedback_prediction').encode('ascii','ignore')

    if db_register.updateFeedback(user_id, video_id, feedback_prediction):
        return jsonify({
            "user_id": user_id,
            "video_id": video_id,
            "prediction": feedback_prediction,
            })
    else:
        return jsonify({"error": 'Entry not found'}),400 # Error 400: Bad Request

    
if __name__ == '__main__':
  
    config_file = 'conf/config_files.yaml'
    configs = Config(config_file)

    os.environ["GOOGLE_APPLICATION_CREDENTIALS"]=configs.application_credentials
    db_register = registerDB.RegisterAnnotations(configs.path_file_annotations)
    with open(configs.dataset_json, 'r') as j:
        dataset_anns = json.loads(j.read())

    app.config['VIDEO_FOLDER'] = configs.video_folder
    app.config['MAX_CONTENT_LENGTH'] = configs.allowed_max_size
    app.config['IPS'] = configs.ip_servers

    print (app.config['IPS'])
    #app.run(host='0.0.0.0', port=56076, debug=True, ssl_context='adhoc')
    context = ssl.SSLContext(ssl.PROTOCOL_TLSv1_2)
    context.load_cert_chain("credentials/cert.pem", "credentials/key.pem")
    app.run(host='0.0.0.0', port=56076, debug=True, use_reloader=False, ssl_context=context)
 
