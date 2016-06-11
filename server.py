# System imports
import subprocess
import time
from os import path
import shutil
import math

#import caffe

import numpy as np
from flask.ext.cors import CORS
from flask import *
from werkzeug import secure_filename

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
        return app.send_static_file("index.html")


@app.route("/dist/<path:asset_path>")
def send_static(asset_path):
    return send_from_directory(static_assets_path, asset_path)


@app.route("/api/upload", methods=["POST"])
def upload():
    def is_allowed(file_name):
        return len(filter(lambda ext: ext in file_name, ["jpg", "png"])) > 0

    image_file = request.files.getlist("video")[0]

    if file and is_allowed(image_file.filename):
        file_name = secure_filename(image_file.filename)
        file_path = path.join(app.config["UPLOAD_FOLDER"], file_name)
        image_file.save(file_path)

        response = jsonify(get_prediction(file_path))
    else:
        response = bad_request("Invalid file")

    return response


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
    pass


def predict_caffe(frame_files):
    net = caffe.Net(app.config["CAFFE_SPATIAL_PROTO"], app.config["CAFFE_SPATIAL_MODEL"], caffe.TEST)
    batch_size = app.config["CAFFE_BATCH_LIMIT"]

    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
    transformer.set_transpose('data', (2, 0, 1))
    transformer.set_raw_scale('data', 255)  # the reference model operates on images in [0,255] range instead of [0,1]
    transformer.set_channel_swap('data', (2, 1, 0))  # the reference model has channels in BGR order instead of RGB
    transformer.set_mean('data', np.load(app.config["CAFFE_SPATIAL_MEAN"]).mean(1).mean(1))  # mean pixel

    caffe.set_mode_cpu()

    results = np.zeros((len(frame_files), app.config["CAFFE_NUM_LABELS"]))

    for cidx, chunk in enumerate(slice_in_chunks(frame_files, batch_size)):

        data = load_frames(chunk, net.blobs['data'].data.shape[2], net.blobs['data'].data.shape[3], transformer)

        net.blobs['data'].reshape(*data.shape)

        out = net.forward_all(data=data)
        print "Finished chunk %d of %d" % (cidx, math.ceil(len(frame_files) * 1.0 / batch_size))
        results[cidx*batch_size: cidx*batch_size+len(chunk), :] = out['prob']
    return results


def get_prediction_caffe(file_path):

    # predictions = external_script.predict(file_path)
    predictions = predict_caffe(frames_in_folder(temp_dir))
    print "Shape of predicitons", predictions.shape, "Type", type(predictions)
    print "Max ", np.argmax(predictions, axis=1)

    file_path += "?cachebuster=%s" % time.time()
    result = {
        "video": {
            "url": "%s" % file_path,
            "framerate": 25
        },
        "frames": []
    }

    return result


if __name__ == "__main__":
    # Start the server
    app.config.update(
        DEBUG=True,
        SECRET_KEY="asassdfs",
        CORS_HEADERS="Content-Type",
        UPLOAD_FOLDER="uploads",
        TEMP_FOLDER="temp",
        CAFFE_BATCH_LIMIT=50,
        CAFFE_NUM_LABELS=101,
        CAFFE_SPATIAL_PROTO="/Users/tombocklisch/Documents/Studium/Master Project/models/deploy.prototxt",
        CAFFE_SPATIAL_MODEL="/Users/tombocklisch/Documents/Studium/Master Project/models/_iter_70000.caffemodel",
        CAFFE_SPATIAL_MEAN="/Users/tombocklisch/Documents/Studium/Master Project/models/ilsvrc_2012_mean.npy"  #"/home/mpss2015/caffe/python/caffe/imagenet/ilsvrc_2012_mean.npy"
    )


    # Start the Flask app
    app.run(port=9000, threaded=True)
