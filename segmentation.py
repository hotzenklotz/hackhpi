from sklearn.cluster import KMeans
import numpy as np
from scipy.ndimage import imread, generic_filter
from scipy.misc import imsave, imresize
from sklearn import linear_model
from skimage.exposure import adjust_gamma, equalize_hist, equalize_adapthist
from skimage.segmentation import clear_border, slic
from skimage.filters import gaussian
from sklearn.externals import joblib 
import sys
import os

class Segmenter():

    def __init__(self):
        self.min_neighbour_count = 3
        self.margin = 20

    def clean_image(self, labeled_img):
        footprint = np.array([[1,1,1],
                              [1,1,1],
                              [1,1,1]])
        def get_new_value(values):
            value = values[4]
            if (values.sum() - value) < self.min_neighbour_count:
                return 0
            else:
                return value   
        
        # return generic_filter(labeled_img, get_new_value, footprint=footprint)
        return (gaussian(clear_border(labeled_img), sigma=1) > 0.99)

    def get_bounding_rect(self, labeled_img):
        cleaned_labeled_img = self.clean_image(labeled_img)
        left = labeled_img.shape[0]
        right = 0
        top = labeled_img.shape[1]
        bottom = 0
        
        for ix, row in enumerate(cleaned_labeled_img):
            for iy, value in enumerate(row):
                if value == 1:
                    if ix < left:
                        left = ix
                    if right < ix:
                        right = ix
                    if iy < top:
                        top = iy
                    if bottom < iy:
                        bottom = iy
        print((top, right, bottom, left))
        return (top, right, bottom, left)



class KMeansSegmenter(Segmenter):
    """Finds mole in image and returns bounding rectangle of it"""

    def __init__(self):
        super().__init__()

    def average_brightness(self, values):
        return np.mean(values)

    def switch_labels(self, labels):
        return np.array([1 - label for label in labels])

    def segment(self, image):
        image = equalize_adapthist(image)
        (x,y,c) = image.shape

        model = KMeans(n_clusters=2)
        points = np.mat(image.reshape(x * y, c))

        model.fit(points)
        labels = model.labels_
        print(labels.sum())
        (center0, center1) = model.cluster_centers_

        print((center0, center1))

        if (self.average_brightness(center0) < self.average_brightness(center1)):
            labels = self.switch_labels(labels)

        labeled_image = labels.reshape(x, y)
        (top, right, bottom, left) = self.get_bounding_rect(labeled_image)
        return image[left:right,top:bottom]


class LogRegSegmenter(Segmenter):
    """docstring for LogRegSegmenter"""
    
    def __init__(self):
        super().__init__()
        self.logreg = self.load_model('model/logreg_model.pkl')

    def load_model(self, pickle_name):
        return joblib.load(pickle_name) 
    
    def segment(self, image):
        img = equalize_adapthist(image)
        (x,y,c) = img.shape
        points = img.reshape(x * y, c)
        labels = self.logreg.predict(points)
        labeled_img = labels.reshape(x, y)
        (top, right, bottom, left) = self.get_bounding_rect(labeled_img)
        # return img[left-self.margin:right+self.margin,top-self.margin:bottom+self.margin]
        return img[left:right,top:bottom]


def test():
    kmeans_segmenter = KMeansSegmenter()
    logreg_segmenter = LogRegSegmenter()
    if len(sys.argv) < 2:
        print("Provide filename")
        exit()
    
    filename = sys.argv[1]
    image = imresize(imread(filename), 50)

    # segmented_image_kmeans  = kmeans_segmenter.segment(image)
    segmented_image_logreg = logreg_segmenter.segment(image)


    # imsave(filename + '_segmented_kmeans.jpg', segmented_image_kmeans)
    imsave(filename + '_segmented_logreg.jpg', segmented_image_logreg)


def segment_dataset():
    kmeans_segmenter = KMeansSegmenter()
    logreg_segmenter = LogRegSegmenter()

    for folder in ['data2/train', 'data2/test']:
        for subfolder in ['atypical', 'melanoma', 'common']:
            if os.path.isdir(os.path.join(folder, subfolder)):
                for file in os.listdir(os.path.join(folder, subfolder)):
                    image = imread(os.path.join(folder, subfolder, file))
                    # segmented_image_kmeans  = kmeans_segmenter.segment(image)
                    segmented_image_logreg = logreg_segmenter.segment(image)

                    # imsave(os.path.join(folder + '_segmented_kmeans', subfolder, file), segmented_image_kmeans)
                    try:
                        imsave(os.path.join(folder + '_segmented_logreg', subfolder, file), segmented_image_logreg)
                    except Exception as e:
                        print(e)
                        pass


if __name__ == "__main__":
    # test()  
    segment_dataset()
