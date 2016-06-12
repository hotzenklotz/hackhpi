import cv2
import numpy as np
import os
import sys
import json
import requests
from multiprocessing import Pool

OUTPUT_DIR = "/Users/therold/Programming/ML/hackhpi/output"

def download_image(image):

  image_id = image["_id"]

  download_url = "https://isic-archive.com:443/api/v1/image/%s/download" % image_id
  metadata_url = "https://isic-archive.com:443/api/v1/image/%s" % image_id

  np_array = np.fromstring(requests.get(download_url).content, np.uint8)
  image = cv2.imdecode(np_array, cv2.IMREAD_COLOR)

  metadata = json.loads(requests.get(metadata_url).text)
  clinical_metdata = metadata["meta"]["clinical"]
  class_name = clinical_metdata.get("benign_malignant", None)

  if not class_name:
    # Class is missing :-(
    return

  # Downscale image if needed
  width, height = image.shape[:2]
  if width > 1024:

    aspect_ratio = float(width) / float(height)

    new_width = 1024
    new_height = int(new_width * aspect_ratio)
    image = cv2.resize(image, (new_width, new_height))

  file_name  = os.path.join(OUTPUT_DIR, class_name, "%s.jpg" % image_id)
  print file_name
  cv2.imwrite(file_name, image)

  return


if __name__ == '__main__':

  images_url = "https://isic-archive.com:443/api/v1/image?limit=250&offset=%s&sort=lowerName&sortdir=1&datasetId=5627f5f69fc3c132be08d852" % sys.argv[1]
  image_ids = json.loads(requests.get(images_url).text)

  for image in image_ids:
    download_image(image)
