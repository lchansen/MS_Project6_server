#!/usr/bin/python

import tornado.web

from tornado.web import HTTPError
from tornado.httpserver import HTTPServer
from tornado.ioloop import IOLoop
from tornado.options import define, options

from basehandler import BaseHandler

import time
import json
import pdb


class MSLC(BaseHandler):
    def get(self):
        self.write('''
            <!DOCTYPE html>
            <html>
            <body>

            <h1>Database Queries</h1>

            ''')
        # now we can display the queries
        # as HTML
        for f in self.db.queries.find():
            self.write('<p>'+str(f)+'</p>')

        self.write('''
            </body>
            </html>
            ''')

class TestHandler(BaseHandler):
    def get(self):
        '''Write out to screen
        '''
        self.write("Test of Hello World")

class PostHandlerAsGetArguments(BaseHandler):
    def post(self):
        ''' If a post request at the specified URL
        Respond with arg1 and arg1*4
        '''
        arg1 = self.get_float_arg("arg1",default=1.0)
        self.write_json({"arg1":arg1,"arg2":4*arg1})

    def get(self):
        '''respond with arg1*2
        '''
        arg1 = self.get_float_arg("arg1",default=3.0);
        # self.write("Get from Post Handler? " + str(arg1*2));
        self.write_json({"arg1":arg1,"arg2":2*arg1})

class JSONPostHandler(BaseHandler):
    def post(self):
        '''Respond with arg1 and arg1*4
        '''
        #print(self.request.body.decode("utf-8"))
        data = json.loads(self.request.body.decode("utf-8"))
        print(data)
        self.write_json({"arg1":data['arg'][0]*2,
            "arg2":data['arg'],
            "arg3":[32,4.5,"Eric Rocks!"]})


class LogToDatabaseHandler(BaseHandler):
    def get(self):
        '''log query to database
        '''
        #pdb.set_trace() # to stop here and inspect
        
        vals = self.get_argument("arg")
        t = time.time()
        ip = self.request.remote_ip
        dbid = self.db.queries.insert(
            {"arg":vals,"time":t,"remote_ip":ip}
            )
        self.write_json({"id":str(dbid)})

# deprecated functionality 
class FileUploadHandler(BaseHandler):
    def post(self):
        print(str(self.request))
        # nginx should be setup for this to work properly
        # you will need to forward the fields to get it running
        # something with _name and _path
