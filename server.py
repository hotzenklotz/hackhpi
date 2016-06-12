# System imports
import subprocess
import time
import os
from os import path
import shutil
import math
import numpy as np
from flask.ext.cors import CORS
from flask import *
from werkzeug import secure_filename
import requests
import json

import env
import segmentation
from scipy.ndimage import imread
from scipy.misc import imsave, imresize

static_assets_path = path.join(path.dirname(__file__))
app = Flask(__name__, static_folder=static_assets_path)
CORS(app)

# ----- Routes ----------
@app.route("/", defaults={"fall_through": ""})
@app.route("/<path:fall_through>")
def index(fall_through):
    if fall_through:
        return bad_request("This url does not exist.")
    else:
        return render_template("home.html")


@app.route("/static/<path:asset_path>")
def send_static(asset_path):
    return send_from_directory(static_assets_path, asset_path)


@app.route("/result")
def result():

  prediction = request.args.get("prediction")
  return render_template("result.html", prediction=prediction)

@app.route("/upload", methods=["POST"])
def upload():
    def is_allowed(file_name):
        return len(filter(lambda ext: ext in file_name, ["jpg", "png"])) > 0

    image_file = request.files["image"]

    if image_file and is_allowed(image_file.filename):
        file_name = secure_filename(image_file.filename)
        file_path = path.join(app.config["UPLOAD_FOLDER"], file_name)
        image_file.save(file_path)

        return redirect("/result?prediction=%s" % get_prediction_ibm(file_path))
    else:
        return bad_request("Invalid file")


@app.route("/api/hubot")
def hubot():
    def is_allowed(file_name):
        return len(filter(lambda ext: ext in file_name, ["jpg", "png"])) > 0

    url = request.args.get("url")
    r = requests.get(url)

    with open('hubot.png', 'wb') as f:
        f.write(r.content)


    return get_prediction_ibm('hubot.png')



def bad_request(reason):
    response = jsonify({"error": reason})
    response.status_code = 400
    return response


# -------- Prediction & Features --------
LABEL_MAPPING = {
    0 : "bad",
    1 : "good",
    2 : "oh_no"
}

def get_prediction_ibm(file_path):

    url = "https://gateway-a.watsonplatform.net/visual-recognition/api/v3/classify?api_key=%s&version=2016-06-11&threshold=0.0" % env.IBM_BLUEMIX_API_KEY
    payload = [
       ('parameters', ('ibm_params.json', open("ibm_params.json", "rb"), 'application/json')),
       ('images_file', ('image.png', preprocess_image(file_path), 'image/png'))
    ]
    response = requests.post(url, files=payload)
    print response.text, type(response.text)
    json_response = json.loads(response.text)

    if json_response.get("error"):
        return bad_request(json_response["error"]["description"])

    classifiers = json_response["images"][0]["classifiers"]

    best_class = None
    best_score = 0
    for class_prediction in classifiers[0]["classes"]:
      if class_prediction["score"] > best_score:
        best_score = class_prediction["score"]
        best_class = class_prediction["class"]

    return best_class


def preprocess_image(file_path):

    image = imread(file_path)
    width, height = image.shape[:2]
    aspect_ratio = float(height) / width
    height = 300
    width = int(aspect_ratio * height)
    image = imresize(image, (height, width))

    out_file = os.path.join("to_server", "foo.png")
    imsave(out_file, image)

    # segmenter = segmentation.KMeansSegmenter()
    # segmented_image = segmenter.segment(image)
    # imsave(out_file, segmented_image)

    return open(out_file, "rb")


if __name__ == "__main__":
    # Start the server
    app.config.update(
        DEBUG=True,
        SECRET_KEY="asassdfs",
        CORS_HEADERS="Content-Type",
        UPLOAD_FOLDER="uploads",
        TEMP_FOLDER="temp",
    )

    if not path.isdir("uploads"):
        os.mkdir("uploads")

    if not path.isdir("to_server"):
        os.mkdir("to_server")

    # Start the Flask app
    app.run(port=9000, threaded=True)
