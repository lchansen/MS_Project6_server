#!/usr/bin/python

from pymongo import MongoClient
import tornado.web

from tornado.web import HTTPError
from tornado.httpserver import HTTPServer
from tornado.ioloop import IOLoop
from tornado.options import define, options

from basehandler import BaseHandler

# model imports
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import pickle
from bson.binary import Binary
import json
import numpy as np

from audioutility import AudioUtility

class PrintHandlers(BaseHandler):
    @tornado.web.authenticated
    def get(self):
        '''Write out to screen the handlers used
        This is a nice debugging example!
        '''
        self.set_header("Content-Type", "application/json")
        self.write(self.application.handlers_string.replace('),','),\n'))

class UploadLabeledDatapointHandler(BaseHandler):
    @tornado.web.authenticated
    def post(self):
        '''Save data point and class label to database
        '''
        self.set_header("Content-Type", "application/json")
        try:
            data = json.loads(self.request.body.decode("utf-8"))
            signal = list(data['signal'])
            sample_rate = int(data['sample_rate'])
            label = str(data['label'])
            dsid  = int(data['dsid'])
        except:
            self.set_status(400) #Bad request
            self.write_json({"status":"invalid request body"})
            return

        try:
            #preprocess the audio, since we are only training the ML model on the mfcc transformation
            signal = [x*10000 for x in signal]
            au = AudioUtility(signal=signal, sample_rate=sample_rate)
            filt, mfcc = au.get_filter_mfcc() #model must be trained on default kwargs
            instance = mfcc.flatten()
            finstance = [float(val) for val in instance] # just in case
            dbid = self.db.labeledinstances.insert({"feature":finstance,"label":label,"dsid":dsid})
        except Exception as e:
            print(e)
            self.set_status(400) #Bad request
            self.write_json({"status":"error processing audio data"})
            return

        self.write_json({"status":"success"})

class RequestNewDatasetId(BaseHandler):
    @tornado.web.authenticated
    def get(self):
        '''Get a new dataset ID for building a new dataset
        '''
        self.set_header("Content-Type", "application/json")
        a = self.db.labeledinstances.find_one(sort=[("dsid", -1)])
        if a == None:
            newSessionId = 1
        else:
            newSessionId = float(a['dsid'])+1
        self.write_json({"status": "success", "dsid":newSessionId})

class UpdateModel(BaseHandler):
    @tornado.web.authenticated
    def post(self):
        '''Train a new model (or update) for given dataset ID
        '''
        self.set_header("Content-Type", "application/json")
        f_res = {"status": "success"}
        try:
            data = json.loads(self.request.body.decode("utf-8")) 
            dsid = data['dsid']
            knn = data['knn']
            svm = data['svm']
        except:
            self.set_status(400) #Bad request
            self.write_json({"status":"invalid request body"})
            return
        
        try:
            # create feature vectors from database
            f=[]
            for a in self.db.labeledinstances.find({"dsid":dsid}): 
                f.append([float(val) for val in a['feature']])

            # create label vector from database
            l=[]
            for a in self.db.labeledinstances.find({"dsid":dsid}): 
                l.append(a['label'])

            if len(set(l)) < 2:
                self.set_status(400) #Bad request
                self.write_json({"status":"Need > class labels"})
                return

            self.models[dsid] = {} # clear current models
            self.models[dsid]['knn'] = KNeighborsClassifier(**data['knn'])
            self.models[dsid]['svm'] = SVC(**data['svm'])

            # fit the model to the data
            acc = {}
            if l:
                for key, clf in self.models[dsid].items():
                    clf.fit(f,l)
                    lstar = clf.predict(f)
                    f_res[key] = str(sum(lstar==l)/float(len(l)))
                    bytes = pickle.dumps(clf)

                    set_obj = {}
                    set_obj[key+'_model'] = Binary(bytes)

                    self.db.models.update(
                        {
                            "dsid":dsid
                        },
                        {  
                            "$set": set_obj
                        },
                        upsert=True)
        except Exception as e:
            self.set_status(400) #Bad request
            self.write_json({"status":"Need > data"})
            return

        # send back the resubstitution accuracy
        # if training takes a while, we are blocking tornado!! No!!
        self.write_json(f_res)

class PredictOne(BaseHandler):
    @tornado.web.authenticated
    def post(self):
        '''Predict the class of a sent feature vector
        '''
        self.set_header("Content-Type", "application/json")
        try:
            data = json.loads(self.request.body.decode("utf-8"))    
            signal = list(data['signal'])
            sample_rate = int(data['sample_rate'])
            dsid  = int(data['dsid'])
            clf_name = data['clf_name']
        except:
            self.set_status(400) #Bad request
            self.write_json({"status":"invalid request body"})
            return

        try:
            #preprocess the audio, since we are only training the ML model on the mfcc transformation
            signal = [x*10000 for x in signal]
            au = AudioUtility(signal=signal, sample_rate=sample_rate)
            filt, mfcc = au.get_filter_mfcc() #model must be trained on default kwargs
            finstance = mfcc.reshape(1,-1) 
        except:
            self.set_status(400) #Bad request
            self.write_json({"status":"error processing audio"})
            return
        

        # load the model from the database if we need to (using pickle)
        # we are blocking tornado!! no!!
        if dsid not in self.models:
            self.models[dsid] = {}
            tmp = self.db.models.find_one({"dsid":dsid})
            if not tmp:
                self.set_status(400) #Bad request
                self.write_json({"status":"No data found for provided DSID"})
                return
            for key in tmp.keys():
                if '_model' in key:
                    self.models[dsid][key[:-6]] = pickle.loads(tmp[key])
        if not self.models[dsid][clf_name]:
            self.set_status(400) #Bad request
            self.write_json({"status":"No records found for the provided classifier"})
            return
        predLabel = self.models[dsid][clf_name].predict(finstance)[0]
        self.write_json({"status": "success", "predLabel":str(predLabel)})
