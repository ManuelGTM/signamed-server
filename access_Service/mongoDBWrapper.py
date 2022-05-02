from pymongo import MongoClient
import numpy as np
import random
import time

MONGO_URI = 'mongodb://localhost'
BD_NAME = 'signamed'
COLLECTION_INFERENCE = 'inference'
COLLECTION_COLLABORATIONS = 'collaborations'
COLLECTION_USERS = 'users'
COLLECTIONS_SIGNS = 'signs'
COLLECTIONS_DATASET_BUILDER = 'dataset-builder'


class Permission:
    INVITED = 0
    REGISTERED = 1
    COLLABORATOR = 2
    ANNOTATOR = 3
    ADMIN = 9


STRUCTURE_ANNS = {'_id': 0, 'user_id': '', 'timestamp': 0,
                  'predictions': [], 'probabilities': [], 'feedback': -1}
STRUCTURE_COLLABORATIONS = {
    '_id': 0, 'user_id': '', 'timestamp': 0, 'sign_id': ''}
STRUCTURE_USER = {'_id': '', 'email': '',
                  'signer_label': '', 'permission': Permission.INVITED}


class SignamedMongoDB():
    def __init__(self):
        self.client = MongoClient(MONGO_URI)
        self.db = self.client[BD_NAME]
        self.inference = self.db[COLLECTION_INFERENCE]
        self.collaborations = self.db[COLLECTION_COLLABORATIONS]
        self.users = self.db[COLLECTION_USERS]
        self.signs = self.db[COLLECTIONS_SIGNS]
        self.dataset_builder = self.db[COLLECTIONS_DATASET_BUILDER]

    def insert_inference(self, videoname: str, user_id: str, timestamp: int, predictions: list, probabilities: list, feedback=-1):
        anns = STRUCTURE_ANNS
        anns['_id'] = videoname
        anns['user_id'] = user_id
        anns['timestamp'] = timestamp
        anns['predictions'] = predictions
        anns['probabilities'] = probabilities
        anns['feedback'] = feedback
        self.inference.insert(anns)

    def update_inference_feedback(self, videoname: str, user_id: str, feedback: int):
        print('inference_update_feedback')
        target = {'_id': videoname, 'user_id': user_id}
        newvalues = {"$set": {'feedback': feedback}}
        if (self.inference.update_one(target, newvalues).matched_count > 0):
            return True
        else:
            return False

    def insert_collaborations(self, videoname, user_id, timestamp, sign_id):
        collaboration = STRUCTURE_COLLABORATIONS
        collaboration['_id'] = videoname
        collaboration['user_id'] = user_id
        collaboration['timestamp'] = timestamp
        collaboration['sign_id'] = sign_id
        self.collaborations.insert(collaboration)

    def get_User_Permission(self, user_id):
        print(self.users.find_one({"_id": user_id}))
        return self.users.find_one({"_id": user_id})['permission']
    
    def get_DasetBuilder_Session(self, session):
        # print(self.dataset_builder.find_one({"session": session}))
        return self.dataset_builder.find_one({"session": session})
    
    def insert_DatasetBuilder_Annotation(self, user_id, session, filename, class_id, annotation_code_id, comment, counter_repeats_used):
        new_anns = {
            "user_id": user_id, 
            "timestamp":  str(int(time.time()*1000000)),
            "class_id":  class_id,
            "annotation_code":  annotation_code_id,
            "comment" : comment,
            "counter_repeats_used": counter_repeats_used
            }
        # position_idx = 8
        # target = {"session" : "prueba", "data.filename": "video8.mp4"}
        # dataset_builder_mongoDB.update_one(target, { "$push": {"data."+str(position_idx)+".inference": new_anns}})
        target = {"session" : session, "data.filename": filename}
        return self.dataset_builder.update_one(target, { "$push": {"data.$.annotations": new_anns}})

    def update_DatasetBuilder_Annotation_Counters(self, session):
        target = {"session" : session}
        for idx, item in enumerate(self.dataset_builder.find_one(target)['data']):
            counter_idx = len(item['annotations'])
            self.dataset_builder.update_one(target, { "$set": {"data."+str(idx)+".counter": counter_idx}})
            
    
    def get_DatasetBuilder_next_Video_Random(self, dataset_builder_session, user_id):
        
            data = dataset_builder_session['data']
            data_idx = data[random.randint(0,len(data)-1)]

            return data_idx    
       
    def get_DatasetBuilder_next_Video_Improved_Method(self, dataset_builder_session, user_id, recent_videos):
        
        data = dataset_builder_session['data']
        # Obtain ids of annotators for each video
        ids = [[annotation['user_id'] for annotation in video['annotations']]
            for video in data]
        # Obtain videos not annotated by user
        user_candidates = [data[i] for i in range(len(data))
                        if user_id not in ids[i]]
        # If all videos were annotated by user, use less annotated videos
        if not user_candidates:
            repetitions = np.array([video_ids.count(user_id) for video_ids in ids])
            selected = repetitions == min(repetitions)
            user_candidates = [data[i] for i in range(len(data)) if selected[i]]
        # Read number of annotations for each candidate video
        counter = np.array([video['counter'] for video in user_candidates])
        # Discard videos with more than N annotations
        prio = counter < 6
        f = max
        # If all videos were annotated more than N times, use all videos
        if (prio.sum() == 0):
            prio = np.ones_like(counter, dtype=bool)
            f = min
        # Select videos with maximum (minimum) number of annotations
        candidates = []
        while not candidates:
            selected = counter == f(counter[prio])
            candidates = [user_candidates[i] for i in range(len(user_candidates))
                        if selected[i] and user_candidates[i]['filename'] not in recent_videos]
            if not candidates:
                prio[counter == f(counter[prio])] = False

        # Select random video from candidates
        data_idx = random.choice(candidates)
        
        return data_idx


if __name__ == '__main__':
    signamedMongoDB = SignamedMongoDB()
    session = 'Componentes No Manuales'
    user_id = 'VpKaNIzpIsae3d2So8wN531cKB52'
    dataset_builder_session = signamedMongoDB.get_DasetBuilder_Session(session)
    data_idx = signamedMongoDB.get_DatasetBuilder_next_Video_Improved_Method(dataset_builder_session, user_id, ['0001.mp4', '0002.mp4', '0003.mp4'])