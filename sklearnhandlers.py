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
        data = json.loads(self.request.body.decode("utf-8"))

        #preprocess the audio, since we are only training the ML model on the mfcc transformation
        au = AudioUtility(signal=data['signal'], sample_rate=data['sample_rate'])
        filt, mfcc = au.get_filter_mfcc() #model must be trained on default kwargs
        instance = mfcc.flatten()
        finstance = [float(val) for val in instance] # just in case
        label = data['label']
        dsid  = data['dsid']

        dbid = self.db.labeledinstances.insert(
            {"feature":finstance,"label":label,"dsid":dsid}
            )
        self.write_json({"id":str(dbid),
                         "feature":[str(len(finstance))+" Points Received",
                         "min of: " +str(min(finstance)),
                         "max of: " +str(max(finstance))],
                         "label":label})

class RequestNewDatasetId(BaseHandler):
    @tornado.web.authenticated
    def get(self):
        '''Get a new dataset ID for building a new dataset
        '''
        a = self.db.labeledinstances.find_one(sort=[("dsid", -1)])
        if a == None:
            newSessionId = 1
        else:
            newSessionId = float(a['dsid'])+1
        self.write_json({"dsid":newSessionId})

class UpdateModel(BaseHandler):
    @tornado.web.authenticated
    def post(self):
        '''Train a new model (or update) for given dataset ID
        '''
        data = json.loads(self.request.body.decode("utf-8")) 
        dsid = data['dsid']

        # create feature vectors from database
        f=[]
        for a in self.db.labeledinstances.find({"dsid":dsid}): 
            f.append([float(val) for val in a['feature']])

        # create label vector from database
        l=[]
        for a in self.db.labeledinstances.find({"dsid":dsid}): 
            l.append(a['label'])

        self.models[dsid] = {} # clear current models
        self.models[dsid]['knn'] = KNeighborsClassifier(**data['knn'])
        self.models[dsid]['svm'] = SVC(**data['svm'])

        # fit the model to the data
        acc = {}
        if l:
            for key, clf in self.models[dsid].items():
                clf.fit(f,l)
                lstar = clf.predict(f)
                acc[key] = sum(lstar==l)/float(len(l))
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

        # send back the resubstitution accuracy
        # if training takes a while, we are blocking tornado!! No!!
        self.write_json({"resubAccuracy":acc})

class PredictOne(BaseHandler):
    @tornado.web.authenticated
    def post(self):
        '''Predict the class of a sent feature vector
        '''
        data = json.loads(self.request.body.decode("utf-8"))    
        dsid = data['dsid']
        clf_name = data['clf_name']

        #preprocess the audio, since we are only training the ML model on the mfcc transformation
        au = AudioUtility(signal=data['signal'], sample_rate=data['sample_rate'])
        filt, mfcc = au.get_filter_mfcc() #model must be trained on default kwargs
        instance = mfcc.flatten()
        finstance = [float(val) for val in instance] # just in case

        # load the model from the database if we need to (using pickle)
        # we are blocking tornado!! no!!
        if dsid not in self.models:
            print('Loading Model From DB')
            tmp = self.db.models.find_one({"dsid":dsid})
            for key in tmp.keys():
                if '_model' in key:
                    self.models[dsid][key[:-6]] = pickle.loads(tmp[key])

        predLabel = self.models[dsid][clf_name].predict(finstance)
        self.write_json({"prediction":str(predLabel)})
