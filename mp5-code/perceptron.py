# perceptron.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Justin Lizama (jlizama2@illinois.edu) on 10/27/2018

"""
This is the main entry point for MP5. You should only modify code
within this file -- the unrevised staff files will be used for all other
files and classes when code is run, so be careful to not modify anything else.
"""

import numpy as np

def classify(train_set, train_labels, dev_set, learning_rate,max_iter):
	"""
	train_set - A Numpy array of 32x32x3 images of shape [7500, 3072].
				This can be thought of as a list of 7500 vectors that are each
				3072 dimensional.  We have 3072 dimensions because there are
				each image is 32x32 and we have 3 color channels.
				So 32*32*3 = 3072
	train_labels - List of labels corresponding with images in train_set
	example: Suppose I had two images [X1,X2] where X1 and X2 are 3072 dimensional vectors
			 and X1 is a picture of a dog and X2 is a picture of an airplane.
			 Then train_labels := [1,0] because X1 contains a picture of an animal
			 and X2 contains no animals in the picture.

	dev_set - A Numpy array of 32x32x3 images of shape [2500, 3072].
			  It is the same format as train_set
	"""
	# TODO: Write your code here
	signLabels = np.zeros(len(train_labels))		# The sign/step labels. +1/-1. Can also be of size len(train_set). Same size.
	weights = np.zeros(train_set.shape[1])			# Initialize weights to zero. A weight for each feature of an image. (3072 in this case).
	bias = 0										# A bias

	# Convert labels to sign/step values
	for labelIndex, label in enumerate(train_labels):
		if label == 0:
			signLabels[labelIndex] = -1				# If it is not an animal, set to -1
		else:
			signLabels[labelIndex] = 1				# If it is an animal, set to +1
	for epoch in range(max_iter):							# Go through the training data however many times max_iter is to refine data.
		permutation = np.random.permutation(train_set.shape[0])				# Shuffle the training set for each epoch
		train_set = train_set[permutation]
		signLabels = signLabels[permutation]
		for imageIndex, image in enumerate(train_set):			# Go through each image.
			result = np.sign(np.dot(weights, image) + bias)
			if result != signLabels[imageIndex]:										# If results not same, we update the weights and bias
				bias = bias + (learning_rate * (signLabels[imageIndex] - result))
				weights = weights + (learning_rate * (signLabels[imageIndex] - result) * image)

	predictedLabels = [] 
	for image in dev_set:
		result = np.sign(np.dot(weights, image) + bias)
		if result == 1:
			predictedLabels.append(1)
		else:
			predictedLabels.append(0)
	# return predicted labels of dev_setlopment set
	return predictedLabels

def classifyEC(train_set, train_labels, dev_set,learning_rate,max_iter):
	# Write your code here if you would like to attempt the extra credit
	return []
