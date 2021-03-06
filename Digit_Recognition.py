
import os

from sklearn import datasets
from sklearn.svm import SVC #For classifying the digits"
from skimage.io import imread
from skimage.exposure import rescale_intensity
from skimage.transform import resize
print("Kézzel írt karakterfelismerő program\n\n")
TEST_IMAGE = input("Kérem adja meg az input fájl elérési útját: ")

    
digits = datasets.load_digits()
features = digits.data
labels = digits.target
    
clf = SVC(gamma = 0.001)
clf.fit(features, labels)

img = resize(imread(os.path.join(TEST_IMAGE)), (8,8)) #Importing the image as float64,resizing the image to 8 by 8
img = rescale_intensity(img, out_range=(0, 16)) 
x_test = [sum(pixel)/3.0 for row in img for pixel in row]
print("The predicted digit is {}".format(clf.predict([x_test])))