#!/usr/bin/python
'''Starts and runs the scikit learn server'''

# For this to run properly, MongoDB must be running
#    Navigate to where mongo db is installed and run
#    something like $./mongod --dbpath "../data/db"
#    might need to use sudo (yikes!)

# database imports
from pymongo import MongoClient
from pymongo.errors import ServerSelectionTimeoutError


# tornado imports
import tornado.web
from tornado.web import HTTPError
from tornado.httpserver import HTTPServer
from tornado.ioloop import IOLoop
from tornado.options import define, options

# custom imports
from basehandler import BaseHandler
import sklearnhandlers as skh
import handlers as hd

# Setup information for tornado class
define("port", default=80, help="run on the given port", type=int)

# Utility to be used when creating the Tornado server
# Contains the handlers and the database connection
class Application(tornado.web.Application):
    def __init__(self):
        '''Store necessary handlers,
           connect to database
        '''
        models = {}
        # format for above object ^
        # {
        #     dsidX: {
        #         'knn': {},
        #         'svm' {},
        #         ...
        #     },
        #     ....
        # }

        handlers = [(r"/[/]?", BaseHandler),
                    (r"/Handlers[/]?",        skh.PrintHandlers,                 dict(models=models)),
                    (r"/AddDataPoint[/]?",    skh.UploadLabeledDatapointHandler, dict(models=models)),
                    (r"/GetNewDatasetId[/]?", skh.RequestNewDatasetId,           dict(models=models)),
                    (r"/UpdateModel[/]?",     skh.UpdateModel,                   dict(models=models)),     
                    (r"/PredictOne[/]?",      skh.PredictOne,                    dict(models=models)),    
                    (r"/Login[/]?",           hd.LoginHandler,                   dict(models=models)),
                    (r"/Logout[/]?",          hd.LogoutHandler,                  dict(models=models)),          
                    ]

        self.handlers_string = str(handlers)

        try:
            self.client  = MongoClient(serverSelectionTimeoutMS=50) # local host, default port
            print(self.client.server_info()) # force pymongo to look for possible running servers, error if none running
            # if we get here, at least one instance of pymongo is running
            self.db = self.client.sklearndatabase # database with labeledinstances, models
            
        except ServerSelectionTimeoutError as inst:
            print('Could not initialize database connection, stopping execution')
            print('Are you running a valid local-hosted instance of mongodb?')
        
        settings = {
            'debug': True,
            'cookie_secret': 'D0N7_U$3_TH!$_1N_PR0D',
            "login_url": "/Authenticate",
        }
        tornado.web.Application.__init__(self, handlers, **settings)

    def __exit__(self):
        self.client.close() # just in case


def main():
    '''Create server, begin IOLoop 
    '''
    tornado.options.parse_command_line()
    http_server = HTTPServer(Application(), xheaders=True)
    http_server.listen(options.port)
    IOLoop.instance().start()

if __name__ == "__main__":
    main()
