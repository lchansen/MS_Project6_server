#!/usr/bin/python

from pymongo import MongoClient
import tornado.web

from tornado.web import HTTPError
from tornado.httpserver import HTTPServer
from tornado.ioloop import IOLoop
from tornado.options import define, options

from basehandler import BaseHandler
import json

class LoginHandler(BaseHandler):
    def get(self):
        self.set_header("Content-Type", "application/json")
        if self.current_user:
            self.write_json({"user": self.current_user})
        else:
            self.write_json({"user": ""})

    def post(self):
        data = json.loads(self.request.body.decode("utf-8")) 
        if 'username' in data and 'password' in data:
            if data['username']=='user' and data['password']=='pass':
                self.set_secure_cookie("user", data['username'])
                self.set_status(200) # OK
                self.finish()
                return
        self.set_status(401) # unauthorized
        self.finish()

class LogoutHandler(BaseHandler):
    @tornado.web.authenticated
    def get(self):
        self.clear_cookie("user")
        self.set_status(200) # OK
        self.finish()
    @tornado.web.authenticated
    def post(self):
        self.clear_cookie("user")
        self.set_status(200) # OK
        self.finish()
