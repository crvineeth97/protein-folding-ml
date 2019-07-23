"""
A module to help in the visualization during validation
Note that the structures built from dihedrals is not very accurate
"""

import threading

from flask import Flask, jsonify, request
from flask_cors import CORS, cross_origin

from constants import HIDE_UI
from util import write_out

app = Flask(__name__)
cors = CORS(app)
data = None


@app.route("/graph", methods=["POST"])
def update_graph():
    global data
    data = request.json
    return jsonify({"result": "OK"})


@app.route("/graph", methods=["GET"])
@cross_origin()
def get_graph():
    return jsonify(data)


@app.route("/", methods=["GET"])
@cross_origin()
def index():
    return open("web/index.html", "r").read()


class graphWebServer(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)

    def run(self):
        import logging

        logging.basicConfig(filename="output/app.log", level=logging.DEBUG)
        app.run(debug=False, host="0.0.0.0")


class frontendWebServer(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)

    def run(self):
        from subprocess import call

        call(["/bin/bash", "start_web_app.sh"])


def start_dashboard_server():
    flask_thread = graphWebServer()
    flask_thread.start()
    front_end_thread = frontendWebServer()
    front_end_thread.start()


def start_visualization():
    if HIDE_UI:
        write_out("Live plot deactivated, see output folder for plot.")
    # Start web server
    # TODO Add more options to view as well as use GDT_TS for scoring
    if not HIDE_UI:
        start_dashboard_server()
