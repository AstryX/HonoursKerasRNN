#RNN Classifier for multi-variable synthetic data set classification and prediction for the Honours Project
#Robertas Dereskevicius 2019/03 University of Dundee
import pandas as pd
import numpy as np
import math as math
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import CuDNNLSTM
from keras.layers import Flatten
from keras.layers import TimeDistributed
from keras.layers import Activation
import matplotlib.pyplot as plt
from scipy import stats
import tensorflow as tf

import statistics


#RNN params 
epochCount = 100
batchSize = 128
hiddenUnitCount = 32
trainingRatio = 0.8

#Multiple of 3
numLinearData = 1500
numLinearDataSingle = int(numLinearData / 3)
#For our sick patient data will get multiplied by 3
numExponentialDecreasing = 675
numExponentialDecreasingFalse = 75
numExponentialIncreasing = 675
numExponentialIncreasingFalse = 75

totalDataLen = numLinearData + 3 * (numExponentialDecreasing + numExponentialDecreasingFalse + numExponentialIncreasing + numExponentialIncreasingFalse)

hdlconst = 8
ldlconst = 8
trigsconst = 13
hba1cconst = 0.4
ubpconst = 10.5

perturbationincrement = 0.25

hdl_value = 80
ldl_value = 80
trigs_value = 130
hba1c_value = 4
ubp_value = 105

majorityelementrange = 20
minorityelementrange = 2

majoritylineardisplacement = [-5, 0, 5]
minoritylineardisplacement = [-0.5, 0, 0.5]

expmajoritysteepness = 0.5
expminoritysteepness = 0.3

initialexpshift = 0.5


def getQuadHDL(i):
	return pow(0.5*((i+1) - 10), 2) + 70

#Similar boundaries as hdl
	
def getQuadLDL(i):
	return pow(0.5*((i+1) - 10), 2) + 70

def getQuadTRIGS(i):
	return pow(0.5*((i+1) - 10), 2) + 120

def getQuadHBA1C(i):
	return pow(0.15*((i+1) - 10), 2) + 3

def getQuadUBP(i):
	return pow(0.5*((i+1 )- 10), 2) + 95		
	

def getExpHDL(i, reverse, offsetcoefficient):
	if reverse == 1:
		return -expmajoritysteepness*pow(1.2, (i+1)) + 70 + majorityelementrange + (majorityelementrange*offsetcoefficient)
	else:
		return expmajoritysteepness*pow(1.2, (i+1)) + 70 - (majorityelementrange*offsetcoefficient)
	

#Similar boundaries as hdl
	
def getExpLDL(i, reverse, offsetcoefficient):
	if reverse == 1:
		return -expmajoritysteepness*pow(1.2, (i+1)) + 70 + majorityelementrange + (majorityelementrange*offsetcoefficient)
	else:
		return expmajoritysteepness*pow(1.2, (i+1)) + 70 - (majorityelementrange*offsetcoefficient)

def getExpTRIGS(i, reverse, offsetcoefficient):
	if reverse == 1:
		return -expmajoritysteepness*pow(1.2, (i+1)) + 120 + majorityelementrange + (majorityelementrange*offsetcoefficient)
	else:
		return expmajoritysteepness*pow(1.2, (i+1)) + 120 - (majorityelementrange*offsetcoefficient)

def getExpHBA1C(i, reverse, offsetcoefficient):
	if reverse == 1:
		return -expminoritysteepness*pow(1.2, (i+1) - 10) + 3 + minorityelementrange + (minorityelementrange*offsetcoefficient)
	else:
		return expminoritysteepness*pow(1.2, (i+1) - 10) + 3 - (minorityelementrange*offsetcoefficient)

def getExpUBP(i, reverse, offsetcoefficient):
	if reverse == 1:
		return -expmajoritysteepness*pow(1.2, (i+1)) + 95 + majorityelementrange + (majorityelementrange*offsetcoefficient)
	else:
		return expmajoritysteepness*pow(1.2, (i+1)) + 95 - (majorityelementrange*offsetcoefficient)
	

def generateSyntheticData(drugType, curveType, dataQuantity, currentParameters, isIncreasing = False, isError = False):
	if curveType == 'Linear' and drugType == 'NoDrug':
		#Does it for 3 times
		displacement = 0
		X_generated = np.zeros((int(dataQuantity * len(majoritylineardisplacement)),20,5))	
		y_generated = []
		for k in range(len(majoritylineardisplacement)):
			for j in range(dataQuantity):
				newseq = np.zeros((20,5))
				for i in range(20):
					perturbed_hdl_value = hdl_value + majoritylineardisplacement[k] + ( np.random.normal(0,1) ) * currentParameters[0][0]
					perturbed_ldl_value = ldl_value + majoritylineardisplacement[k] + ( np.random.normal(0,1) ) * currentParameters[1][0]
					perturbed_trigs_value = trigs_value + majoritylineardisplacement[k] + ( np.random.normal(0,1) ) * currentParameters[2][0]
					perturbed_hba1c_value = hba1c_value + minoritylineardisplacement[k] + ( np.random.normal(0,1) ) * currentParameters[3][0]
					perturbed_ubp_value = ubp_value + majoritylineardisplacement[k] + ( np.random.normal(0,1) ) * currentParameters[4][0]
					newseq[i][0] = perturbed_hdl_value
					newseq[i][1] = perturbed_ldl_value
					newseq[i][2] = perturbed_trigs_value
					newseq[i][3] = perturbed_hba1c_value
					newseq[i][4] = perturbed_ubp_value
				y_generated.append([1,0,0,0])
				X_generated[j+displacement] = newseq
				
			displacement = displacement + dataQuantity
			
		return X_generated, y_generated	
		
	elif curveType == 'Exponential':
		X_generated = np.zeros((dataQuantity,20,5))	
		y_generated = []
		
		if drugType == 'Drug1Drug2':
			if isError == True:
				positionalMajorityShift = 0
				positionalMinorityShift = 0
				if isIncreasing == True:
					positionalMajorityShift = -(majorityelementrange)
					positionalMinorityShift = -(minorityelementrange)
				elif isIncreasing == False:
					positionalMajorityShift = majorityelementrange
					positionalMinorityShift = minorityelementrange
				#no response
				for j in range(dataQuantity):
					newseq = np.zeros((20,5))
					for i in range(20):
						perturbed_hdl_value = hdl_value + positionalMajorityShift + ( np.random.normal(0,1) ) * currentParameters[0][0]
						perturbed_ldl_value = ldl_value + positionalMajorityShift + ( np.random.normal(0,1) ) * currentParameters[1][0]
						perturbed_trigs_value = trigs_value + positionalMajorityShift + ( np.random.normal(0,1) ) * currentParameters[2][0]
						perturbed_hba1c_value = hba1c_value + positionalMinorityShift + ( np.random.normal(0,1) ) * currentParameters[3][0]
						perturbed_ubp_value = ubp_value + positionalMajorityShift + ( np.random.normal(0,1) ) * currentParameters[4][0]
						newseq[i][0] = perturbed_hdl_value
						newseq[i][1] = perturbed_ldl_value
						newseq[i][2] = perturbed_trigs_value
						newseq[i][3] = perturbed_hba1c_value
						newseq[i][4] = perturbed_ubp_value
					y_generated.append([0,0,0,1])
					X_generated[j] = newseq
			elif isError == False:
				isReverse = -1
				positionalMajorityShift = 0
				positionalMinorityShift = 0
				if isIncreasing == True:
					isReverse = 0
					positionalMajorityShift = -(initialexpshift * majorityelementrange)
					positionalMinorityShift = -(initialexpshift * minorityelementrange)
				elif isIncreasing == False:
					isReverse = 1
					positionalMajorityShift = (initialexpshift * majorityelementrange)
					positionalMinorityShift = (initialexpshift * minorityelementrange)
				#Quadratic increasing with response
				for j in range(dataQuantity):
					newseq = np.zeros((20,5))
					for i in range(20):
						hdl_value_quad = getExpHDL(i,isReverse,currentParameters[0][1])
						ldl_value_quad = getExpLDL(i,isReverse,currentParameters[1][1])
						trigs_value_quad = getExpTRIGS(i,isReverse,currentParameters[2][1])
						hba1c_value_quad = getExpHBA1C(i,isReverse,currentParameters[3][1])
						ubp_value_quad = getExpUBP(i,isReverse,currentParameters[4][1])
						perturbed_hdl_value = hdl_value_quad + positionalMajorityShift + ( np.random.normal(0,1) ) * currentParameters[0][0]
						perturbed_ldl_value = ldl_value_quad + positionalMajorityShift + ( np.random.normal(0,1) ) * currentParameters[1][0]
						perturbed_trigs_value = trigs_value_quad + positionalMajorityShift + ( np.random.normal(0,1) ) * currentParameters[2][0]
						perturbed_hba1c_value = hba1c_value_quad + positionalMinorityShift + ( np.random.normal(0,1) ) * currentParameters[3][0]
						perturbed_ubp_value = ubp_value_quad + positionalMajorityShift + ( np.random.normal(0,1) ) * currentParameters[4][0]
						newseq[i][0] = perturbed_hdl_value
						newseq[i][1] = perturbed_ldl_value
						newseq[i][2] = perturbed_trigs_value
						newseq[i][3] = perturbed_hba1c_value
						newseq[i][4] = perturbed_ubp_value
					y_generated.append([0,0,0,1])
					X_generated[j] = newseq	
		
		elif drugType == 'Drug1':
			if isError == True:
				positionalMajorityShift = 0
				positionalMinorityShift = 0
				if isIncreasing == True:
					positionalMajorityShift = -(majorityelementrange)
					positionalMinorityShift = -(minorityelementrange)
				elif isIncreasing == False:
					positionalMajorityShift = majorityelementrange
					positionalMinorityShift = minorityelementrange
				#no response
				for j in range(dataQuantity):
					newseq = np.zeros((20,5))
					for i in range(20):
						perturbed_hdl_value = hdl_value + positionalMajorityShift + ( np.random.normal(0,1) ) * currentParameters[0][0]
						perturbed_ldl_value = ldl_value + positionalMajorityShift + ( np.random.normal(0,1) ) * currentParameters[1][0]
						perturbed_trigs_value = trigs_value + ( np.random.normal(0,1) ) * currentParameters[2][0]
						perturbed_hba1c_value = hba1c_value + ( np.random.normal(0,1) ) * currentParameters[3][0]
						perturbed_ubp_value = ubp_value + ( np.random.normal(0,1) ) * currentParameters[4][0]
						newseq[i][0] = perturbed_hdl_value
						newseq[i][1] = perturbed_ldl_value
						newseq[i][2] = perturbed_trigs_value
						newseq[i][3] = perturbed_hba1c_value
						newseq[i][4] = perturbed_ubp_value
					y_generated.append([0,1,0,0])
					X_generated[j] = newseq
			elif isError == False:
				isReverse = -1
				positionalMajorityShift = 0
				positionalMinorityShift = 0
				if isIncreasing == True:
					isReverse = 0
					positionalMajorityShift = -(initialexpshift * majorityelementrange)
					positionalMinorityShift = -(initialexpshift * minorityelementrange)
				elif isIncreasing == False:
					isReverse = 1
					positionalMajorityShift = (initialexpshift * majorityelementrange)
					positionalMinorityShift = (initialexpshift * minorityelementrange)
				#Quadratic increasing with response
				for j in range(dataQuantity):
					newseq = np.zeros((20,5))
					for i in range(20):
						hdl_value_quad = getExpHDL(i,isReverse,currentParameters[0][1])
						ldl_value_quad = getExpLDL(i,isReverse,currentParameters[1][1])
						perturbed_hdl_value = hdl_value_quad + positionalMajorityShift + ( np.random.normal(0,1) ) * currentParameters[0][0]
						perturbed_ldl_value = ldl_value_quad + positionalMajorityShift + ( np.random.normal(0,1) ) * currentParameters[1][0]
						perturbed_trigs_value = trigs_value + ( np.random.normal(0,1) ) * currentParameters[2][0]
						perturbed_hba1c_value = hba1c_value + ( np.random.normal(0,1) ) * currentParameters[3][0]
						perturbed_ubp_value = ubp_value + ( np.random.normal(0,1) ) * currentParameters[4][0]
						newseq[i][0] = perturbed_hdl_value
						newseq[i][1] = perturbed_ldl_value
						newseq[i][2] = perturbed_trigs_value
						newseq[i][3] = perturbed_hba1c_value
						newseq[i][4] = perturbed_ubp_value
					y_generated.append([0,1,0,0])
					X_generated[j] = newseq				
		elif drugType == 'Drug2':
			if isError == True:
				positionalMajorityShift = 0
				positionalMinorityShift = 0
				if isIncreasing == True:
					positionalMajorityShift = -(majorityelementrange)
					positionalMinorityShift = -(minorityelementrange)
				elif isIncreasing == False:
					positionalMajorityShift = majorityelementrange
					positionalMinorityShift = minorityelementrange
				#no response
				for j in range(dataQuantity):
					newseq = np.zeros((20,5))
					for i in range(20):
						perturbed_hdl_value = hdl_value + ( np.random.normal(0,1) ) * currentParameters[0][0]
						perturbed_ldl_value = ldl_value + ( np.random.normal(0,1) ) * currentParameters[1][0]
						perturbed_trigs_value = trigs_value + positionalMajorityShift + ( np.random.normal(0,1) ) * currentParameters[2][0]
						perturbed_hba1c_value = hba1c_value + positionalMinorityShift + ( np.random.normal(0,1) ) * currentParameters[3][0]
						perturbed_ubp_value = ubp_value + positionalMajorityShift + ( np.random.normal(0,1) ) * currentParameters[4][0]
						newseq[i][0] = perturbed_hdl_value
						newseq[i][1] = perturbed_ldl_value
						newseq[i][2] = perturbed_trigs_value
						newseq[i][3] = perturbed_hba1c_value
						newseq[i][4] = perturbed_ubp_value
					y_generated.append([0,0,1,0])
					X_generated[j] = newseq
			elif isError == False:
				isReverse = -1
				positionalMajorityShift = 0
				positionalMinorityShift = 0
				if isIncreasing == True:
					isReverse = 0
					positionalMajorityShift = -(initialexpshift * majorityelementrange)
					positionalMinorityShift = -(initialexpshift * minorityelementrange)
				elif isIncreasing == False:
					isReverse = 1
					positionalMajorityShift = (initialexpshift * majorityelementrange)
					positionalMinorityShift = (initialexpshift * minorityelementrange)
				#Quadratic increasing with response
				for j in range(dataQuantity):
					newseq = np.zeros((20,5))
					for i in range(20):
						trigs_value_quad = getExpTRIGS(i,isReverse,currentParameters[2][1])
						hba1c_value_quad = getExpHBA1C(i,isReverse,currentParameters[3][1])
						ubp_value_quad = getExpUBP(i,isReverse,currentParameters[4][1])
						perturbed_hdl_value = hdl_value + ( np.random.normal(0,1) ) * currentParameters[0][0]
						perturbed_ldl_value = ldl_value + ( np.random.normal(0,1) ) * currentParameters[1][0]
						perturbed_trigs_value = trigs_value_quad + positionalMajorityShift + ( np.random.normal(0,1) ) * currentParameters[2][0]
						perturbed_hba1c_value = hba1c_value_quad + positionalMinorityShift + ( np.random.normal(0,1) ) * currentParameters[3][0]
						perturbed_ubp_value = ubp_value_quad + positionalMajorityShift + ( np.random.normal(0,1) ) * currentParameters[4][0]
						newseq[i][0] = perturbed_hdl_value
						newseq[i][1] = perturbed_ldl_value
						newseq[i][2] = perturbed_trigs_value
						newseq[i][3] = perturbed_hba1c_value
						newseq[i][4] = perturbed_ubp_value
					y_generated.append([0,0,1,0])
					X_generated[j] = newseq			
		return X_generated, y_generated
		
	return None,None
	
def trainRNNClassifier(X_train,y_train):

	# create the model
	model = Sequential()
	model.add(CuDNNLSTM(hiddenUnitCount, input_shape=(20,5), return_sequences=True))
	model.add(CuDNNLSTM(hiddenUnitCount, return_sequences=True))
	#model.add(Flatten())
	#model.add(Dense(1, activation='sigmoid'))
	#model.add(Dense(4, activation='softmax'))
	#model.add(TimeDistributed(Dense(1, activation='sigmoid')))
	model.add(CuDNNLSTM(4, return_sequences=False))
	model.add(Activation('sigmoid'))
	model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
	
	model.fit(X_train, y_train, batch_size=batchSize, epochs=epochCount, verbose=0, shuffle=False)
	
	return model
	
def evaluateRNNClassifier(X_set, y_set, foldCount):
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

		X_train= np.reshape(X_train,(trainingCount, 20, 5))
		
		X_test= np.reshape(X_test,(len(X_set) - trainingCount, 20, 5))
		
		model = trainRNNClassifier(X_train, y_train)
		
		scores = model.evaluate(X_test,y_test,verbose=0)
		
		y_New = model.predict_classes(X_test)
		with open('your_file.txt', 'w') as f:
			for item in y_New:
				f.write("%s\n" % item)
		
		'''# re-define model to predict/evaluate on a different batch size
		n_batch = 1
		new_model = Sequential()
		new_model.add(CuDNNLSTM(32, batch_input_shape=(n_batch,20,6), return_sequences=True, stateful=True))
		new_model.add(Flatten())
		new_model.add(Dense(1, activation='sigmoid'))
		
		# copy weights
		old_weights = model.get_weights()
		new_model.set_weights(old_weights)
		
		new_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
		
		predictedValues = []
		
		for testID in range(len(X_test)):
			# Final evaluation of the model
			scores = new_model.predict(np.reshape(X_test[testID],(1,20,6)), verbose=0)
			print(scores)
			predictedValues.append(scores)
			#("Printing test label values:")
			#print(y_test)
		'''
		print("Accuracy: %.2f%%" % (scores[1]*100))
		
		#-----------------------------------print(model.metrics_names)
		
		print(scores)
		#print(predictedValues[0])
		#print("Param nr:"+str(plotcounter)+"; Fold nr:" + str(fold))
		lossarray.append(scores[0])
		accarray.append(scores[1])
		
	return lossarray,accarray
	
def preprocessRNNClassifier(paramVector):
	plotcounter = 0

	for curparamseq in paramvector:
		totalSize = 0
		#syntheticDataset = np.zeros((totalDataLen,20,5))	
		#labels = np.zeros((totalDataLen))
		syntheticDataset = []
		labels = []

		tempX, tempY = generateSyntheticData('NoDrug', 'Linear', numLinearDataSingle, curparamseq)
		syntheticDataset.extend(tempX)
		labels.extend(tempY)
			
		tempX, tempY = generateSyntheticData('Drug1', 'Exponential', numExponentialIncreasing, curparamseq, True, False)
		syntheticDataset.extend(tempX)
		labels.extend(tempY)
		
		tempX, tempY = generateSyntheticData('Drug1', 'Exponential', numExponentialIncreasingFalse, curparamseq, True, True)
		syntheticDataset.extend(tempX)
		labels.extend(tempY)
		
		tempX, tempY = generateSyntheticData('Drug1', 'Exponential', numExponentialDecreasing, curparamseq, False, False)
		syntheticDataset.extend(tempX)
		labels.extend(tempY)
		
		tempX, tempY = generateSyntheticData('Drug1', 'Exponential', numExponentialDecreasingFalse, curparamseq, False, True)
		syntheticDataset.extend(tempX)
		labels.extend(tempY)

		tempX, tempY = generateSyntheticData('Drug2', 'Exponential', numExponentialIncreasing, curparamseq, True, False)
		syntheticDataset.extend(tempX)
		labels.extend(tempY)
		
		tempX, tempY = generateSyntheticData('Drug2', 'Exponential', numExponentialIncreasingFalse, curparamseq, True, True)
		syntheticDataset.extend(tempX)
		labels.extend(tempY)
		
		tempX, tempY = generateSyntheticData('Drug2', 'Exponential', numExponentialDecreasing, curparamseq, False, False)
		syntheticDataset.extend(tempX)
		labels.extend(tempY)
		
		tempX, tempY = generateSyntheticData('Drug2', 'Exponential', numExponentialDecreasingFalse, curparamseq, False, True)
		syntheticDataset.extend(tempX)
		labels.extend(tempY)
			
		tempX, tempY = generateSyntheticData('Drug1Drug2', 'Exponential', numExponentialIncreasing, curparamseq, True, False)
		syntheticDataset.extend(tempX)
		labels.extend(tempY)
		
		tempX, tempY = generateSyntheticData('Drug1Drug2', 'Exponential', numExponentialIncreasingFalse, curparamseq, True, True)
		syntheticDataset.extend(tempX)
		labels.extend(tempY)
		
		tempX, tempY = generateSyntheticData('Drug1Drug2', 'Exponential', numExponentialDecreasing, curparamseq, False, False)
		syntheticDataset.extend(tempX)
		labels.extend(tempY)
		
		tempX, tempY = generateSyntheticData('Drug1Drug2', 'Exponential', numExponentialDecreasingFalse, curparamseq, False, True)
		syntheticDataset.extend(tempX)
		labels.extend(tempY)
		
		lossarray,accarray = evaluateRNNClassifier(syntheticDataset, labels, 5);
		
		avgloss = sum(lossarray) / float(len(lossarray))
		avgacc = sum(accarray) / float(len(accarray))
		standiv = statistics.stdev(accarray)

		print("Average Loss")
		print(avgloss)
		print("Average Acc")
		print(avgacc)
		print("SD of Acc")
		print(standiv)

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
		plt.ylabel('HDL levels')
		plt.xlabel('NO of visit')
		plt.axis([1, 20, 50, 100])
		plt.show()'''
		
		plt.axis([0, 15, 0, 1])
		plt.scatter(plotcounter, avgacc)
		plt.pause(0.05)
		plotcounter = plotcounter + 1
	plt.show()
	
'''def trainRNNRegressor(X_train,y_train):
	
	
def preprocessRNNRegressor(X_train,y_train):
	
#Used for the regressor problem that needs different labels
def constructSyntheticDataLabels(X_train):'''
	
	
paramvector = []

for it in range(4):
	paramseq = []

	paramseq.append([ hdlconst * (perturbationincrement * (it+1)), it * 0.25 ] )
	paramseq.append([ ldlconst * (perturbationincrement * (it+1)), it * 0.25 ] )
	paramseq.append([ trigsconst * (perturbationincrement * (it+1)), it * 0.25 ] )
	paramseq.append([ hba1cconst * (perturbationincrement * (it+1)), it * 0.25 ] )
	paramseq.append([ ubpconst * (perturbationincrement * (it+1)), it * 0.25 ] )

	paramvector.append(paramseq)

preprocessRNNClassifier(paramvector)
