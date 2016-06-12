from sklearn.cluster import KMeans
import numpy as np
from scipy.ndimage import imread, generic_filter
from scipy.misc import imsave, imresize
import sys
import os

class KMeansSegmenter():
    """Finds mole in image and returns bounding rectangle of it"""

    def __init__(self):
        self.min_neighbour_count = 3
        self.margin = 20

    def average_brightness(self, values):
        return np.mean(values)

    def switch_labels(self, labels):
        return np.array([1 - label for label in labels])

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
        
        return generic_filter(labeled_img, get_new_value, footprint=footprint)

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

    def segment(self, image):
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


def test():
    segmenter = KMeansSegmenter()
    if len(sys.argv) < 2:
        print("Provide filename")
        exit()
    
    filename = sys.argv[1]
    image = imresize(imread(filename), 50)
    segmented_image = segmenter.segment(image)
    imsave(filename + "-segmented.jpeg", segmented_image)

def segment_dataset():
    segmenter = KMeansSegmenter()
    for folder in ['data/train', 'data/test']:
        for subfolder in ['benign', 'malignant']:
            for file in os.listdir(os.path.join(folder, subfolder)):
                image = imread(os.path.join(folder, subfolder, file))
                segmented_image  = segmenter.segment(image)
                imsave(os.path.join(folder + '_segmented', subfolder, file), segmented_image)

if __name__ == "__main__":
    # test()  
    segment_dataset()
