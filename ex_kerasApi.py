# import the necessary packages
from keras.applications import ResNet50
from keras.applications import MobileNet
import matplotlib.pyplot as plt

import numpy as np
import cv2
import dlv

if __name__ == "__main__":
	test = np.random.rand(500,300)
	cv2.imshow('img',test)
	cv2.waitKey(0)
	
	
	
	resnet50Model = MobileNet(weights="imagenet")
	dlvModel = dlv.Model(resnet50Model)
	dlvModel.addInputData('dog.jpg')
	dlvModel.getFeaturesFromFetchedList()
	
	result = dlvModel._indata_FeatureMap_Dict['dog.jpg']._featureMapList[0]
	
	result = np.moveaxis(result, -1, 0)
	filter0 = result[0]
	
	# filter0 = filter0.astype(np.uint8)
	cv2.imshow('image',filter0)
	cv2.waitKey(0)
	
	print('tmp')
	
	


#
# if __name__ == "__main__":
# 	resnet50Model = MobileNet(weights="imagenet")
# 	dlvModel = dlv.Model(resnet50Model)
# 	dlvModel.addInputData('dog.jpg')
# 	dlvModel.addInputData('cat.jpg')
# 	dlvModel.getFeaturesFromFetchedList()
#
# 	# Prepare pyplot
#
# 	w = 112
# 	h = 112
#
# 	# fig = plt.figure(figsize=(64, len(dlvModel._indata_FeatureMap_Dict['cat.jpg']._featureMapList)))
# 	fig = plt.figure(figsize=(64, 1))
#
# 	columns = 1
# 	rows = 64
#
# 	for j in range(0, columns):
# 		conv_j_result = dlvModel._indata_FeatureMap_Dict['cat.jpg']._featureMapList[j]
#
# 		for i in range(0, rows ):
# 			subplot = fig.add_subplot(j+1, 64, j*64 + i + 1)
# 			subplot.set_xticks([])
# 			subplot.set_yticks([])
# 			image = conv_j_result[:, :, i]
# 			subplot.imshow(image)
# 	plt.show()
