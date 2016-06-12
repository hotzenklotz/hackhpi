import sys
import os
import env
import requests
import json
import math


BATCH_SIZE = 10

def main(data_root):

  classes = ["atypical", "common", "melanoma"]

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

      immages, classes, scores = zip(*predicted_classes)

      correct_predictions = [predicted_class
                             for predicted_class in classes
                             if predicted_class == correct_class]
      num_correct_class += len(correct_predictions)
      num_correct_global += len(correct_predictions)

      best_score = 0
      best_image = None
      for (image, p_class, score) in predicted_classes:
        if p_class == correct_class:
          if score > best_score:
            best_score = score
            best_image = image

      print("Best in batch: %s, %f" % (best_image, best_score))


    print("Class accuracy: %f" % (float(num_correct_class) / num_class))

  print("Global accuracy: %f" % (float(num_correct_global) / num_global))


def get_predictions(path, batch):

  url = "https://gateway-a.watsonplatform.net/visual-recognition/api/v3/classify?api_key=%s&version=2016-06-11&threshold=0.0" % env.IBM_BLUEMIX_API_KEY
  payload = [
    ('parameters', ('ibm_params.json', open("ibm_params.json", "rb"), 'application/json')),
  ]

  for image in batch:
    payload.append(('images_file', (image, open(os.path.join(path, image), "rb"), 'image/jpeg')))

  response = requests.post(url, files=payload)
  result = json.loads(response.text)

  predicted_classes = []
  for image_result in result["images"]:
    image_name = image_result["image"]
    best_class = None
    best_score = 0
    for class_prediction in image_result["classifiers"][0]["classes"]:
      if class_prediction["score"] > best_score:
        best_score = class_prediction["score"]
        best_class = class_prediction["class"]

    predicted_classes.append((image_name, best_class, best_score))
  return predicted_classes


if __name__ == "__main__":
  main(sys.argv[1])