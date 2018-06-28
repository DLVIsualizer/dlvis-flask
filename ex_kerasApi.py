# import the necessary packages
from keras.applications import ResNet50
import matplotlib.pyplot as plt

import dlv

if __name__ == "__main__":
	resnet50Model = ResNet50(weights="imagenet")
	dlvModel = dlv.Model(resnet50Model)
	dlvModel.addInputData('dog.jpg')
	dlvModel.addInputData('cat.jpg')
	dlvModel.getFeaturesFromFetchedList()
	
	# Prepare pyplot
	
	w = 112
	h = 112
	
	# fig = plt.figure(figsize=(64, len(dlvModel._indata_FeatureMap_Dict['cat.jpg']._featureMapList)))
	fig = plt.figure(figsize=(64, 1))
	
	columns = 1
	rows = 64
	
	for j in range(0, columns):
		conv_j_result = dlvModel._indata_FeatureMap_Dict['cat.jpg']._featureMapList[j]
		
		for i in range(0, rows ):
			subplot = fig.add_subplot(j+1, 64, j*64 + i + 1)
			subplot.set_xticks([])
			subplot.set_yticks([])
			image = conv_j_result[:, :, i]
			subplot.imshow(image)
	plt.show()
