import sys
import os
import env
import requests
import json
import math


BATCH_SIZE = 10

def main(data_root):

  classes = ["benign", "malignant"]

  num_correct_global = 0
  num_global = 0

  for correct_class in classes:

    files = [f for f in os.listdir(os.path.join(data_root, correct_class))
                     if not f.startswith(".")]

    print("Class: %s (%d)" % (correct_class, len(files)))

    num_global += len(files)
    num_correct_class = 0
    num_class = len(files)

    num_batches = int(math.ceil(float(len(files)) / BATCH_SIZE))
    for batch in range(num_batches):
      start_index = BATCH_SIZE * batch
      end_index = min(len(files), BATCH_SIZE * batch + BATCH_SIZE)
      predicted_classes = get_predictions(os.path.join(data_root, correct_class),
                                          files[start_index:end_index])
      correct_predictions = [predicted_class
                             for predicted_class in predicted_classes
                             if predicted_class == correct_class]

      num_correct_class += len(correct_predictions)
      num_correct_global += len(correct_predictions)

    print("Class accuracy: %d" % (float(num_correct_class) / num_class))

  print("Global accuracy: %d" % (float(num_correct_global) / num_global))


def get_predictions(path, batch):

  url = "https://gateway-a.watsonplatform.net/visual-recognition/api/v3/classify?api_key=%s&version=2016-06-11&threshold=0.0" % env.IBM_BLUEMIX_API_KEY
  payload = [
    ('parameters', ('ibm_params.json', open("ibm_params.json", "rb"), 'application/json')),
  ]

  for image in batch:
    payload.append(('images_file', (image, open(os.path.join(path, image), "rb"), 'image/png')))

  response = requests.post(url, files=payload)
  result = json.loads(response.text)

  predicted_classes = []
  for image_result in result["images"]:
    predicted_classes.append(image_result["classifiers"][0]["classes"][0]["class"])
  return predicted_classes


if __name__ == "__main__":
  main(sys.argv[1])