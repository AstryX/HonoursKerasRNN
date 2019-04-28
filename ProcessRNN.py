#RNN Classifier for multi-variable synthetic data set classification and prediction for the Honours Project
#Robertas Dereskevicius 2019/03 University of Dundee
import numpy as np
import math as math
import json
import SimulateData
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import CuDNNLSTM
from keras.layers import Flatten
from keras.layers import TimeDistributed
from sklearn.metrics import confusion_matrix
from keras.layers import Activation
from keras.callbacks import Callback
from keras.models import model_from_yaml
from keras import backend
import sklearn.metrics
import matplotlib.pyplot as plt
import tensorflow as tf
from scipy import stats
import sys
import statistics
import time

#Class that is used for callbacks from neural network training to display real-time graphs of accuracy and/or loss
class DynamicPlot(Callback):
	def __init__(self, title, ylabel, title2 = '', ylabel2 = ''):
		self.dataTitle = title
		self.yLabel = ylabel
		self.dataTitle2 = title2
		self.yLabel2 = ylabel2
	
	def on_train_begin(self, history={}):
		self.iterator = 1
		self.epoch = []
		self.rmse = []
		self.metric2 = []
		plt.close()
		self.fig = plt.figure()
		self.history = []

	#Called after each epoch finishes by the callback
	def on_epoch_end(self, epoch, history={}):   
		self.history.append(history)
		self.epoch.append(self.iterator)
		self.rmse.append(history.get(self.dataTitle))
		if self.dataTitle2 != '':
			self.metric2.append(history.get(self.dataTitle2))
		
		plt.plot(self.epoch, self.rmse, 'g', label = self.yLabel)
		if self.dataTitle2 != '':
			plt.plot(self.epoch, self.metric2, 'r', label = self.yLabel2)
			
		if self.iterator == 1:
			plt.legend()
		plt.ion()
		plt.ylabel('Metrics')
		plt.xlabel('Epoch')
		
		self.iterator += 1
		
		plt.show();
		plt.pause(0.001)	

#Custom root mean squared error function calculation
def rmse(y_real, y_pred):
    return backend.sqrt(backend.mean(backend.square(y_pred - y_real), axis=-1))
	
#Primary function to train a RNN classifier
def trainRNNClassifier(X_train,y_train):

	# create the model
	model = Sequential()
	model.add(Dense(hiddenUnitCount, input_shape=(numTimesteps,numFeatures)))
	model.add(CuDNNLSTM(hiddenUnitCount, return_sequences=True))
	model.add(CuDNNLSTM(hiddenUnitCount))
	model.add(Dense(4, activation='softmax'))
	
	model.compile(loss='sparse_categorical_crossentropy',
            metrics=['sparse_categorical_accuracy'],
			optimizer='adam')
	
	if shouldShowGraph == True:
		plotCallback = DynamicPlot('sparse_categorical_accuracy','Sparse Categorical Accuracy', 'loss', 'Loss')
		model.fit(X_train, y_train, batch_size=batchSize, epochs=epochCount, callbacks=[plotCallback], verbose=0, shuffle=False)
	else:
		model.fit(X_train, y_train, batch_size=batchSize, epochs=epochCount, verbose=0, shuffle=False)
	return model
	
#RNN Classifier testing function
def testRNNClassifier(X_set, y_set, folds, paramCounter):
	accarray = []
	lossarray = []
	confarray = []

	#Runs for a number of cross-validation folds
	for fold in range(folds):
	
		print('Fold count: %d' % fold)
		rng_state = np.random.get_state()
		np.random.shuffle(X_set)
		np.random.set_state(rng_state)
		np.random.shuffle(y_set)
		
		trainingCount = int(trainingRatio * len(X_set))
		
		X_train = X_set[:trainingCount]
		y_train = y_set[:trainingCount]
		X_test = X_set[trainingCount:]
		y_test = y_set[trainingCount:]

		X_train= np.reshape(X_train,(trainingCount, numTimesteps, numFeatures))
		X_test= np.reshape(X_test,(len(X_set) - trainingCount, numTimesteps, numFeatures))
		
		model = trainRNNClassifier(X_train, y_train)
		
		scores = model.evaluate(X_test,y_test,verbose=0)
		
		y_New = model.predict_classes(X_test)
		
		print("Accuracy: %.2f%%" % (scores[1]*100))

		conf_matrix = confusion_matrix((y_test.astype(int)), y_New)
		print(conf_matrix)
		
		print(scores)
		lossarray.append(scores[0])
		accarray.append(scores[1])
		confarray.append(conf_matrix)
		
	if shouldSaveModel == True:
		# serialize model to YAML
		model_yaml = model.to_yaml()
		with open("modelclassifier"+str(paramCounter)+".yaml", "w") as yaml_file:
			yaml_file.write(model_yaml)
		# serialize weights to HDF5
		model.save_weights("modelclassifier"+str(paramCounter)+".h5")
		print("Successfully saved the model.")
		
	return lossarray,accarray,confarray
	
#Main function for RNN classifier that handles data retrieval, parameter iteration and output
def mainRNNClassifier(paramVector, dataGen):
	plotcounter = 0
	
	for curparamseq in paramVector:
		start_t = time.time()
		#That is where data is generated for classification
		syntheticDataset, labels = dataGen.retrieveConcatenatedDrugData(curparamseq)
		
		lossarray,accarray,confarray = testRNNClassifier(syntheticDataset, labels, foldCount, plotcounter);
		
		avgloss = sum(lossarray) / float(len(lossarray))
		avgacc = sum(accarray) / float(len(accarray))
		standiv = statistics.stdev(accarray)
		elapsed_t = time.time() - start_t
		
		#Calculating average values for each confusion matrix member
		avgconf = [[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]]
		for i in range(len(confarray)):
			for j in range(len(confarray[i])):
				avgconf[j][0] += confarray[i][j][0]
				avgconf[j][1] += confarray[i][j][1]
				avgconf[j][2] += confarray[i][j][2]
				avgconf[j][3] += confarray[i][j][3]
			
		for i in range(len(avgconf)):
			avgconf[i][0] = math.floor(avgconf[i][0]/len(confarray))
			avgconf[i][1] = math.floor(avgconf[i][1]/len(confarray))
			avgconf[i][2] = math.floor(avgconf[i][2]/len(confarray))
			avgconf[i][3] = math.floor(avgconf[i][3]/len(confarray))

		print("Average Loss")
		print(avgloss)
		print("Average Acc")
		print(avgacc)
		print("SD of Acc")
		print(standiv)
		#A simple confusion matrix format
		print("Average Confusion Matrix")
		print(avgconf[0])
		print(avgconf[1])
		print(avgconf[2])
		print(avgconf[3])
		
		if plotcounter == 0:
			openType = 'w'
		else:
			openType = 'a'
		
		#Output to a file the results of current parameter iteration
		with open('ClassificationResults.txt', openType) as f:
			f.write("------------------------------------------\n")
			f.write("Classification results for param nr:%d\n" % plotcounter)
			f.write("------------------------------------------\n")
			f.write("Parameter noise strength (HDL):%f\n" % curparamseq[0][0])
			f.write("Parameter offset:%f\n" % curparamseq[0][1])
			f.write("Parameter steepness:%f\n" % curparamseq[0][2])
			f.write("Number of folds:%d\n" % foldCount)
			f.write("------------------------------------------\n")
			f.write("Elapsed time: %s\n" % (time.strftime("%H:%M:%S", time.gmtime(elapsed_t))))
			f.write("Average Loss:%f\n" % avgloss)
			f.write("Average Accuracy:%f\n" % avgacc)
			f.write("SD of Accuracy:%f\n" % standiv)
			f.write("Averaged Confusion Matrix (Floored):\n")
			f.write("  0 1 2 3\n")
			f.write("0 %d %d %d %d\n" % (avgconf[0][0],avgconf[0][1],avgconf[0][2],avgconf[0][3]))
			f.write("1 %d %d %d %d\n" % (avgconf[1][0],avgconf[1][1],avgconf[1][2],avgconf[1][3]))
			f.write("2 %d %d %d %d\n" % (avgconf[2][0],avgconf[2][1],avgconf[2][2],avgconf[2][3]))
			f.write("3 %d %d %d %d\n" % (avgconf[3][0],avgconf[3][1],avgconf[3][2],avgconf[3][3]))
			f.write("\n")
			
		#This commented scatter graph could be used to track param set accuracy in real-time
		'''plt.axis([0, 15, 0, 1])
		plt.scatter(plotcounter, avgacc)
		plt.pause(0.05)'''
		
		plotcounter = plotcounter + 1	
	
#This generator is used for fitting a RNN Regression model
def variableSizeGenerator(X_train, y_train):
	batch_it = 0
	maxBatch = len(X_train) / batchSize
	
	#Runs infinitely until generator stops calling it.
	while True:
		if batch_it == maxBatch:
			batch_it = 0
		
		batch_X = X_train[batch_it * batchSize:batch_it * batchSize + batchSize]
		batch_Y = y_train[batch_it * batchSize:batch_it * batchSize + batchSize]
		
		experimental_X = batch_X[0]
		
		#Have to perform reshaping and reconstruction of an array to format it in the way that a neural network will accept. Numpy weirdness.
		for i in range(len(batch_X)):
			if i != 0:
				experimental_X = np.concatenate((experimental_X,batch_X[i]), axis=0)
		
		batch_X = np.reshape(experimental_X, (batchSize, len(batch_X[0]), numFeatures))
		
		#Returns a batch of features and labels for each call by the generator
		yield batch_X, batch_Y
		batch_it = batch_it + 1
	
#Function that constructs and trains a RNN Regression model
def trainRNNRegressor(X_train,y_train,epochSteps, shouldUseMini):

	# create the model
	model = Sequential()
	model.add(Dense(hiddenUnitCount, input_shape=(None, numFeatures)))
	model.add(CuDNNLSTM(hiddenUnitCount, return_sequences=True))
	model.add(CuDNNLSTM(hiddenUnitCount))
	model.add(Dense(numFeatures))
	model.compile(optimizer='adam', loss="mean_squared_error", metrics=["mean_squared_error", rmse])
	
	if shouldUseMini == True:
		#Special fit_generator function is called for training the network to feed it chunks of data at a time manually
		if shouldShowGraph == True:
			plotCallback = DynamicPlot('rmse','Root Mean Squared Error','mean_squared_error','Mean Squared Error')
			result = model.fit_generator(variableSizeGenerator(X_train, y_train), steps_per_epoch=epochSteps, epochs=epochCount, callbacks=[plotCallback], verbose=0, shuffle=False)
		else:
			result = model.fit_generator(variableSizeGenerator(X_train, y_train), steps_per_epoch=epochSteps, epochs=epochCount, verbose=0, shuffle=False)
	else:
		if shouldShowGraph == True:
			plotCallback = DynamicPlot('rmse','Root Mean Squared Error','mean_squared_error','Mean Squared Error')
			result = model.fit(X_train, y_train, batch_size=batchSize, epochs=epochCount, callbacks=[plotCallback], verbose=0, shuffle=False)
		else:
			result = model.fit(X_train, y_train, batch_size=batchSize, epochs=epochCount, verbose=0, shuffle=False)		

	return model
	
#Function that tests a RNN Regression model and performs cross-validation
def testRNNRegressor(X_regressor, y_regressor, folds, shouldRNNUseMiniBatch, paramCounter):
	lossarray = []

	for fold in range(folds):

		print('Fold count: %d' % fold)
		rng_state = np.random.get_state()
		np.random.shuffle(X_regressor)
		np.random.set_state(rng_state)
		np.random.shuffle(y_regressor)
		
		stepsPerEpoch = None
		
		#Regression data that uses mini-batch boolean has a different structure and must be reconstructed before processing
		if shouldRNNUseMiniBatch == True:
			reconstructed_X = []
			reconstructed_Y = []
			
			for i in range(len(X_regressor)):
				reconstructed_X.extend(X_regressor[i])
				reconstructed_Y.extend(y_regressor[i])
		
			trainingCount = int(trainingRatio * len(reconstructed_X))
			batchRemainder = trainingCount % batchSize
			trainingCount = trainingCount - batchRemainder
			
			X_train = np.array(reconstructed_X[:trainingCount])
			y_train = np.array(reconstructed_Y[:trainingCount])
			X_test = np.array(reconstructed_X[trainingCount:])
			y_test = np.array(reconstructed_Y[trainingCount:])
			
			stepsPerEpoch = int(len(X_train)/batchSize)
		else:
			trainingCount = int(trainingRatio * len(X_regressor))
			
			X_train = np.array(X_regressor[:trainingCount])
			y_train = np.array(y_regressor[:trainingCount])
			X_test = np.array(X_regressor[trainingCount:])
			y_test = np.array(y_regressor[trainingCount:])
		
		
		model = trainRNNRegressor(X_train, y_train, stepsPerEpoch, shouldRNNUseMiniBatch)
		
		#Have to format predictions and run them iteratively for mini-batches due to numpy weirdness
		if shouldRNNUseMiniBatch == True:
			y_predicted = []
			
			for j in range(int(len(X_test)/batchSize)):
				batch_X = X_test[j * batchSize:j * batchSize + batchSize]
				
				experimental_X = batch_X[0]
				
				for z in range(len(batch_X)):
					if z != 0:
						experimental_X = np.concatenate((experimental_X,batch_X[z]), axis=0)
				
				batch_X = np.reshape(experimental_X, (batchSize, len(batch_X[0]), numFeatures))
				y_predicted.extend(model.predict(batch_X))
			
			finalRMSE = math.sqrt(sklearn.metrics.mean_squared_error(y_test,y_predicted))
			print("Root mean squared error of test set(RMSE): %f" % finalRMSE)
			
			lossarray.append(finalRMSE)
		else:
			y_predicted = model.predict(np.reshape(X_test, (len(X_test),numTimesteps-1,5)))
			
			finalRMSE = math.sqrt(sklearn.metrics.mean_squared_error(y_test,y_predicted))
			print("Root mean squared error of test set(RMSE): %f" % finalRMSE)
			
			lossarray.append(finalRMSE)
	
	if shouldSaveModel == True:
		# serialize model to YAML
		model_yaml = model.to_yaml()
		with open("modelregressor"+str(paramCounter)+".yaml", "w") as yaml_file:
			yaml_file.write(model_yaml)
		# serialize weights to HDF5
		model.save_weights("modelregressor"+str(paramCounter)+".h5")
		print("Successfully saved the model.")
	
	return lossarray
	
#Primary function that handles parameter iteration, data generation and formatting, and output for a RNN Regressor
def mainRNNRegressor(paramvector, dataGen):
	plotcounter = 0

	for curparamseq in paramvector:
	
		start_t = time.time()
	
		syntheticDataset, labels = dataGen.retrieveConcatenatedDrugData(curparamseq)
	
		#Extra function call compared to classifier as we need different labels for regression
		X_regressor,y_regressor = dataGen.constructSyntheticDataPredictions(syntheticDataset, shouldRNNUseZeroPadding, shouldRNNUseMiniBatch, batchSize)
		
		lossarray = testRNNRegressor(X_regressor, y_regressor, foldCount, shouldRNNUseMiniBatch, plotcounter);
		
		avgrmse = sum(lossarray) / float(len(lossarray))
		standiv = statistics.stdev(lossarray)
		
		print("Average RMSE")
		print(avgrmse)
		print("SD of RMSE")
		print(standiv)
		
		if plotcounter == 0:
			openType = 'w'
		else:
			openType = 'a'
		
		elapsed_t = time.time() - start_t
		
		with open('RegressionResults.txt', openType) as f:
			f.write("------------------------------------------\n")
			f.write("Regression results for param nr:%d\n" % plotcounter)
			f.write("------------------------------------------\n")
			f.write("Parameter noise strength (HDL):%f\n" % curparamseq[0][0])
			f.write("Parameter offset:%f\n" % curparamseq[0][1])
			f.write("Parameter steepness:%f\n" % curparamseq[0][2])
			f.write("Number of folds:%d\n" % foldCount)
			f.write("------------------------------------------\n")
			f.write("Elapsed time: %s\n" % (time.strftime("%H:%M:%S", time.gmtime(elapsed_t))))
			f.write("Average Root Mean Squared Error:%f\n" % avgrmse)
			f.write("SD of RMSE:%f\n" % standiv)
			f.write("\n")
		
		plotcounter = plotcounter + 1
	
#Function is functional, but the student did not have enough time to implement model load and custom non-generated data set use functionality
def loadRNNModel(path, path2):

	# load the YAML and read the model from it
	yamlModel = open(path, 'r')
	readModel = yamlModel.read()
	yamlModel.close()
	loadedModel = model_from_yaml(readModel)
	# load the read weights into a model
	loadedModel.load_weights(path2)

	return loadedModel
	
#Default global internal values
epochCount = 100
batchSize = 32
hiddenUnitCount = 32
trainingRatio = 0.8
numFeatures = 5	
foldCount = 10
numTimesteps = 20
shouldRNNUseMiniBatch = True
shouldRNNUseZeroPadding = False
isClassifierEnabled = True
isRegressorEnabled = False
shouldShowGraph = True
shouldSaveModel = False
	
if len(sys.argv) > 2:

	with open(sys.argv[1]) as json_data_file:
		data = json.load(json_data_file)
		
		#Global variables loaded from the RNN config
		epochCount = data['epochCount']
		batchSize = data['batchSize']
		hiddenUnitCount = data['hiddenUnitCount']
		trainingRatio = data['trainingRatio']
		foldCount = data['foldCount']
		shouldRNNUseMiniBatch = data['useMiniBatch']
		shouldRNNUseZeroPadding = data['useZeroPadding']
		isClassifierEnabled = data['classifierEnabled']
		isRegressorEnabled = data['regressorEnabled']
		shouldShowGraph = data['enableEpochGraph']
		shouldSaveModel = data['saveModel']

	#Simulate Data object instantiation
	dataGenerator = SimulateData.DataGenerator(sys.argv[2])
	
	#RNN params generated
	globalParamVector = dataGenerator.createParamVector()	
	
	if(isClassifierEnabled == True):
		mainRNNClassifier(globalParamVector, dataGenerator)
		
	if(isRegressorEnabled == True):
		mainRNNRegressor(globalParamVector, dataGenerator)
		
	#Edge case
	if(isClassifierEnabled == False and isRegressorEnabled == False):
		print('Classifier and regressor have been disabled. No neural network operations could be performed. Change the RNN parameter settings.')

else:
	print("Did not provide sufficient arguments! Try running python ProcessRNN.py \"rnnconfig.json\" \"paramconfig.json\"")