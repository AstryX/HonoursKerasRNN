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

class DynamicPlot(Callback):
	def __init__(self, title, ylabel):
		self.dataTitle = title
		self.yLabel = ylabel
	
	def on_train_begin(self, history={}):
		self.iterator = 0
		self.epoch = []
		self.rmse = []
		plt.close()
		self.fig = plt.figure()
		self.history = []

	def on_epoch_end(self, epoch, history={}):   
		self.history.append(history)
		self.epoch.append(self.iterator)
		self.rmse.append(history.get(self.dataTitle))
		self.iterator += 1
			
		#clear_output(wait=True)
		#plt.plot(self.epoch, self.rmse, label="Root Mean Squared Error")
		plt.plot(self.epoch, self.rmse)
		plt.ion()
		#plt.legend()
		plt.ylabel(self.yLabel)
		plt.xlabel('Epoch')
		#plt.axis([0, epochCount, 0, self.rmse[0]])
		
		plt.show();
		plt.pause(0.001)

def rmse(y_real, y_pred):
    return backend.sqrt(backend.mean(backend.square(y_pred - y_real), axis=-1))
	
def trainRNNClassifier(X_train,y_train):

	# create the model
	model = Sequential()
	model.add(Dense(hiddenUnitCount, input_shape=(20,numFeatures)))
	model.add(CuDNNLSTM(hiddenUnitCount, return_sequences=True))
	model.add(CuDNNLSTM(hiddenUnitCount))
	#model.add(Flatten())
	model.add(Dense(4, activation='softmax'))
	#model.add(TimeDistributed(Dense(1, activation='sigmoid')))
	#model.add(CuDNNLSTM(4, return_sequences=False))
	#model.add(Activation('sigmoid'))
	model.compile(loss='sparse_categorical_crossentropy',
            metrics=['sparse_categorical_accuracy'],
			optimizer='adam')
	
	plotCallback = DynamicPlot('sparse_categorical_accuracy','Sparse Categorical Accuracy')
	
	model.fit(X_train, y_train, batch_size=batchSize, epochs=epochCount, callbacks=[plotCallback], verbose=0, shuffle=False)
	
	return model
	
def testRNNClassifier(X_set, y_set, foldCount):
	accarray = []
	lossarray = []

	for fold in range(foldCount):
		rng_state = np.random.get_state()
		np.random.shuffle(X_set)
		np.random.set_state(rng_state)
		np.random.shuffle(y_set)
		
		trainingCount = int(trainingRatio * len(X_set))
		
		
		
		X_train = X_set[:trainingCount]
		y_train = y_set[:trainingCount]
		X_test = X_set[trainingCount:]
		y_test = y_set[trainingCount:]

		X_train= np.reshape(X_train,(trainingCount, 20, numFeatures))
		X_test= np.reshape(X_test,(len(X_set) - trainingCount, 20, numFeatures))
		
		model = trainRNNClassifier(X_train, y_train)

		scores = model.evaluate(X_test,y_test,verbose=0)
		
		y_New = model.predict_classes(X_test)
		with open('your_file.txt', 'w') as f:
			for item in y_New:
				f.write("%s\n" % item)
		
		print("Accuracy: %.2f%%" % (scores[1]*100))
		
		#-----------------------------------print(model.metrics_names)
		
		print(scores)
		#print(predictedValues[0])
		#print("Param nr:"+str(plotcounter)+"; Fold nr:" + str(fold))
		lossarray.append(scores[0])
		accarray.append(scores[1])
		
	return lossarray,accarray
	
def mainRNNClassifier(paramVector, dataGen):
	plotcounter = 0
	
	for curparamseq in paramVector:

		syntheticDataset, labels = dataGen.retrieveConcatenatedDrugData(curparamseq)
		
		'''index = []
		for i in range(20):
			index.append(i+1)

		#slope, intercept, r_value, p_value, std_err = stats.linregress(index,hdllist)
		slope, intercept, r_value, p_value, std_err = stats.linregress(index,syntheticDataset[2800,:20,0])

		linear = []	
		for i in range(20):
			linear.append(slope * index[i] + intercept)
			
			
		#plt.plot(index, hdllist, index, linear)
		plt.plot(index, syntheticDataset[2800,:20,0], index, linear)
		plt.xticks(index)
		plt.ylabel('Simulated HDL levels with noise')
		plt.xlabel('NO of visit')
		plt.axis([1, 20, 70, 110])
		plt.show()'''
		
		lossarray,accarray = testRNNClassifier(syntheticDataset, labels, 5);
		
		avgloss = sum(lossarray) / float(len(lossarray))
		avgacc = sum(accarray) / float(len(accarray))
		standiv = statistics.stdev(accarray)

		print("Average Loss")
		print(avgloss)
		print("Average Acc")
		print(avgacc)
		print("SD of Acc")
		print(standiv)
		
		'''plt.axis([0, 15, 0, 1])
		plt.scatter(plotcounter, avgacc)
		plt.pause(0.05)
		plotcounter = plotcounter + 1'''
	plt.show()	
	
def variableSizeGenerator(X_train, y_train):
	batch_it = 0
	maxBatch = len(X_train) / batchSize
	while True:
		if batch_it == maxBatch:
			batch_it = 0
		
		batch_X = X_train[batch_it * batchSize:batch_it * batchSize + batchSize]
		batch_Y = y_train[batch_it * batchSize:batch_it * batchSize + batchSize]
		
		experimental_X = batch_X[0]
		
		for i in range(len(batch_X)):
			if i != 0:
				experimental_X = np.concatenate((experimental_X,batch_X[i]), axis=0)
		
		batch_X = np.reshape(experimental_X, (batchSize, len(batch_X[0]), numFeatures))
		
		yield batch_X, batch_Y
		batch_it = batch_it + 1
	

def trainRNNRegressor(X_train,y_train,epochSteps):

	# create the model
	model = Sequential()
	model.add(Dense(hiddenUnitCount, input_shape=(None, numFeatures)))
	model.add(CuDNNLSTM(hiddenUnitCount, return_sequences=True))
	model.add(CuDNNLSTM(hiddenUnitCount))
	'''model.add(CuDNNLSTM(hiddenUnitCount, input_shape=(None, 2), return_sequences=True))'''
	
	model.add(Dense(numFeatures))
	model.compile(optimizer='adam', loss="mean_squared_error", metrics=["mean_squared_error", rmse])
	
	plotCallback = DynamicPlot('rmse','Root Mean Squared Error')
	
	result = model.fit_generator(variableSizeGenerator(X_train, y_train), steps_per_epoch=epochSteps, epochs=epochCount, callbacks=[plotCallback], verbose=1, shuffle=False)
	
	return model
	
def testRNNRegressor(X_regressor, y_regressor, foldCount, shouldRNNUseMiniBatch):
	lossarray = []

	for fold in range(foldCount):

		rng_state = np.random.get_state()
		np.random.shuffle(X_regressor)
		np.random.set_state(rng_state)
		np.random.shuffle(y_regressor)
		
		stepsPerEpoch = None
		
		if shouldRNNUseMiniBatch == True:
			reconstructed_X = []
			reconstructed_Y = []
			
			stepsPerEpoch = len(X_regressor)
			
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
		else:
			trainingCount = int(trainingRatio * len(X_regressor))
			
			X_train = np.array(X_regressor[:trainingCount])
			y_train = np.array(y_regressor[:trainingCount])
			X_test = np.array(X_regressor[trainingCount:])
			y_test = np.array(y_regressor[trainingCount:])
		
		
		model = trainRNNRegressor(X_train, y_train, stepsPerEpoch)

		'''#X_data = np.array(([[[1],[2]],[[1],[2],[3]],[[1],[2],[3],[5]],[[1],[2],[3],[5],[8]]]))
		#y_data = [3,5,8,13]
		
		X_data = np.array([[[1,1],[2,1.5],[3,2.5],[5,4]],[[2,1.5],[3,2.5],[5,4],[8,6.5]],[[3,2.5],[5,4],[8,6.5],[13,10.5]],[[5,4],[8,6.5],[13,10.5],[21,17]]])
		y_data = np.array([[8,6.5],[13,10.5],[21,17],[34,27.5]])
		
		x = np.random.randn(1,7,1)
		y = np.random.randn(1)
		
		model = trainRNNRegressor(X_data,y_data)
		
		testx = np.array([[[1,1],[2,1.5],[3,2.5],[5,4]]])
		#testx = np.reshape(testx,(1, 4, 1))
		
		ytest = model.predict(testx)
		print('Output of regressor:')
		print(ytest)'''
		ytest = model.predict(np.reshape(X_test[20],(1,19,5)))
		print('Input to reg')
		print(X_test[20])
		print('Output of regressor:')
		print(ytest)
		print('Real output')
		print(y_test[20])
		
		y_predicted = model.predict(np.reshape(X_test, (len(X_test),19,5)))
		
		#print("Mean squared error (MSE):       %f" % sklearn.metrics.mean_squared_error(y_test,y_predicted))
		print("Root mean squared error of test set(RMSE): %f" % math.sqrt(sklearn.metrics.mean_squared_error(y_test,y_predicted)))
		
		lossarray.append(math.sqrt(sklearn.metrics.mean_squared_error(y_test,y_predicted)))
	
	# serialize model to YAML
	model_yaml = model.to_yaml()
	with open("model.yaml", "w") as yaml_file:
		yaml_file.write(model_yaml)
	# serialize weights to HDF5
	model.save_weights("model.h5")
	print("Successfully saved the model.")
	
	return lossarray
	
def mainRNNRegressor(paramvector, dataGen):
	for curparamseq in paramvector:

		shouldRNNUseMiniBatch = True
	
		syntheticDataset, labels = dataGen.retrieveConcatenatedDrugData(curparamseq)
	
		X_regressor,y_regressor = dataGen.constructSyntheticDataPredictions(syntheticDataset, False, shouldRNNUseMiniBatch, batchSize)
		
		lossarray = testRNNRegressor(X_regressor, y_regressor, 5, shouldRNNUseMiniBatch);
		
		avgrmse = sum(lossarray) / float(len(lossarray))

		print("Average RMSE")
		print(avgrmse)
	

def loadRNNModel(path):

	# load the YAML and read the model from it
	yamlModel = open(path, 'r')
	readModel = yamlModel.read()
	yamlModel.close()
	loadedModel = model_from_yaml(readModel)
	# load the read weights into a model
	loadedModel.load_weights("model.h5")

	return loadedModel
	
epochCount = 100
batchSize = 128
hiddenUnitCount = 32
trainingRatio = 0.8
numFeatures = 5	
	
if len(sys.argv) > 2:

	with open(sys.argv[1]) as json_data_file:
			data = json.load(json_data_file)
			
			epochCount = data['epochCount']
			batchSize = data['batchSize']
			hiddenUnitCount = data['hiddenUnitCount']
			trainingRatio = data['trainingRatio']
			numFeatures = data['numFeatures']

	#RNN params 
	dataGenerator = SimulateData.DataGenerator(sys.argv[2])

	globalParamVector = dataGenerator.createParamVector(4)	
	mainRNNClassifier(globalParamVector, dataGenerator)
	#mainRNNRegressor(paramvector, dataGenerator)

else:
	print("Did not provide sufficient arguments! Try running python ProcessRNN.py \"rnnconfig.json\" \"paramconfig.json\"")