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
import matplotlib.pyplot as plt
from scipy import stats

import statistics


#RNN params 
epochCount = 100
batchSize = 128


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
	

def trainRNNClassifier(X_set, y_set, foldCount):
	accarray = []
	lossarray = []

	for fold in range(foldCount):
		# create the model
		model = Sequential()
		model.add(CuDNNLSTM(32, batch_input_shape=(600,20,5), return_sequences=True, stateful=True))
		#model.add(CuDNNLSTM(32, return_sequences=True, stateful=True))
		model.add(Flatten())
		model.add(Dense(1, activation='sigmoid'))
		#model.add(TimeDistributed(Dense(1, activation='sigmoid')))
		model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
		
		rng_state = np.random.get_state()
		np.random.shuffle(X_set)
		np.random.set_state(rng_state)
		np.random.shuffle(y_set)
		
		X_train = X_set[:2400]
		y_train = y_set[:2400]
		X_test = X_set[2400:]
		y_test = y_set[2400:]
		
		X_test= np.reshape(X_test,(600, 20, 5))

		# truncate and pad input sequences
		X_train= np.reshape(X_train,(2400, 20, 5))
		
		#model.fit(X_train, y_train, batch_size=400, epochs=100, verbose=0)
		for bNumber in range(4):
			model.fit(X_train[(bNumber*600):(bNumber*600 + 600)], y_train[(bNumber*600):(bNumber*600 + 600)], batch_size=600, epochs=epochCount, verbose=0, shuffle=False)
		
		scores = model.evaluate(X_test,y_test,batch_size=600, verbose=0)
		
		
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
		
		#reset the states so the model does not carry info over
		model.reset_states()
		
	return lossarray,accarray
	
def preprocessRNNClassifier(paramVector):
	plotcounter = 0

	for curparamseq in paramvector:
		totalSize = 0
		generate_quadratic = np.zeros((3000,20,5))	
		labels = np.zeros((3000))

		for k in range(len(majoritylineardisplacement)):
			for j in range(500):
				newseq = np.zeros((20,5))
				for i in range(20):
					perturbed_hdl_value = hdl_value + majoritylineardisplacement[k] + ( np.random.normal(0,1) ) * curparamseq[0][0]
					perturbed_ldl_value = ldl_value + majoritylineardisplacement[k] + ( np.random.normal(0,1) ) * curparamseq[1][0]
					perturbed_trigs_value = trigs_value + majoritylineardisplacement[k] + ( np.random.normal(0,1) ) * curparamseq[2][0]
					perturbed_hba1c_value = hba1c_value + minoritylineardisplacement[k] + ( np.random.normal(0,1) ) * curparamseq[3][0]
					perturbed_ubp_value = ubp_value + majoritylineardisplacement[k] + ( np.random.normal(0,1) ) * curparamseq[4][0]
					newseq[i][0] = perturbed_hdl_value
					newseq[i][1] = perturbed_ldl_value
					newseq[i][2] = perturbed_trigs_value
					newseq[i][3] = perturbed_hba1c_value
					newseq[i][4] = perturbed_ubp_value
				labels[j+totalSize] = 0
				generate_quadratic[j+totalSize] = newseq

			totalSize = totalSize + 500
			
			
		#Quadratic increasing with response
		for j in range(675):
			newseq = np.zeros((20,5))
			for i in range(20):
				hdl_value_quad = getExpHDL(i,0,curparamseq[0][1])
				ldl_value_quad = getExpLDL(i,0,curparamseq[1][1])
				trigs_value_quad = getExpTRIGS(i,0,curparamseq[2][1])
				hba1c_value_quad = getExpHBA1C(i,0,curparamseq[3][1])
				ubp_value_quad = getExpUBP(i,0,curparamseq[4][1])
				perturbed_hdl_value = hdl_value_quad - (initialexpshift * majorityelementrange) + ( np.random.normal(0,1) ) * curparamseq[0][0]
				perturbed_ldl_value = ldl_value_quad - (initialexpshift * majorityelementrange) + ( np.random.normal(0,1) ) * curparamseq[1][0]
				perturbed_trigs_value = trigs_value_quad - (initialexpshift * majorityelementrange) + ( np.random.normal(0,1) ) * curparamseq[2][0]
				perturbed_hba1c_value = hba1c_value_quad - (initialexpshift * minorityelementrange) + ( np.random.normal(0,1) ) * curparamseq[3][0]
				perturbed_ubp_value = ubp_value_quad - (initialexpshift * majorityelementrange) + ( np.random.normal(0,1) ) * curparamseq[4][0]
				newseq[i][0] = perturbed_hdl_value
				newseq[i][1] = perturbed_ldl_value
				newseq[i][2] = perturbed_trigs_value
				newseq[i][3] = perturbed_hba1c_value
				newseq[i][4] = perturbed_ubp_value
			labels[j+totalSize] = 1
			generate_quadratic[j+totalSize] = newseq

		totalSize = totalSize + 675
		
		#no response
		for j in range(75):
			newseq = np.zeros((20,5))
			for i in range(20):
				perturbed_hdl_value = hdl_value - majorityelementrange + ( np.random.normal(0,1) ) * curparamseq[0][0]
				perturbed_ldl_value = ldl_value - majorityelementrange + ( np.random.normal(0,1) ) * curparamseq[1][0]
				perturbed_trigs_value = trigs_value - majorityelementrange + ( np.random.normal(0,1) ) * curparamseq[2][0]
				perturbed_hba1c_value = hba1c_value - minorityelementrange + ( np.random.normal(0,1) ) * curparamseq[3][0]
				perturbed_ubp_value = ubp_value - majorityelementrange + ( np.random.normal(0,1) ) * curparamseq[4][0]
				newseq[i][0] = perturbed_hdl_value
				newseq[i][1] = perturbed_ldl_value
				newseq[i][2] = perturbed_trigs_value
				newseq[i][3] = perturbed_hba1c_value
				newseq[i][4] = perturbed_ubp_value
			labels[j+totalSize] = 1
			generate_quadratic[j+totalSize] = newseq

		totalSize = totalSize + 75
			
		#Quadratic decreasing with response
		for j in range(675):
			newseq = np.zeros((20,5))
			for i in range(20):
				hdl_value_quad = getExpHDL(i,1,curparamseq[0][1])
				ldl_value_quad = getExpLDL(i,1,curparamseq[1][1])
				trigs_value_quad = getExpTRIGS(i,1,curparamseq[2][1])
				hba1c_value_quad = getExpHBA1C(i,1,curparamseq[3][1])
				ubp_value_quad = getExpUBP(i,1,curparamseq[4][1])
				perturbed_hdl_value = hdl_value_quad + (initialexpshift * majorityelementrange) + ( np.random.normal(0,1) ) * curparamseq[0][0]
				perturbed_ldl_value = ldl_value_quad + (initialexpshift * majorityelementrange) + ( np.random.normal(0,1) ) * curparamseq[1][0]
				perturbed_trigs_value = trigs_value_quad + (initialexpshift * majorityelementrange) + ( np.random.normal(0,1) ) * curparamseq[2][0]
				perturbed_hba1c_value = hba1c_value_quad + (initialexpshift * minorityelementrange) + ( np.random.normal(0,1) ) * curparamseq[3][0]
				perturbed_ubp_value = ubp_value_quad + (initialexpshift * majorityelementrange) + ( np.random.normal(0,1) ) * curparamseq[4][0]
				newseq[i][0] = perturbed_hdl_value
				newseq[i][1] = perturbed_ldl_value
				newseq[i][2] = perturbed_trigs_value
				newseq[i][3] = perturbed_hba1c_value
				newseq[i][4] = perturbed_ubp_value
			labels[j+totalSize] = 1
			generate_quadratic[j+totalSize] = newseq		
			
		totalSize = totalSize + 675
		
		#no response
		for j in range(75):
			newseq = np.zeros((20,5))
			for i in range(20):
				perturbed_hdl_value = hdl_value + majorityelementrange + ( np.random.normal(0,1) ) * curparamseq[0][0]
				perturbed_ldl_value = ldl_value + majorityelementrange + ( np.random.normal(0,1) ) * curparamseq[1][0]
				perturbed_trigs_value = trigs_value + majorityelementrange + ( np.random.normal(0,1) ) * curparamseq[2][0]
				perturbed_hba1c_value = hba1c_value + minorityelementrange + ( np.random.normal(0,1) ) * curparamseq[3][0]
				perturbed_ubp_value = ubp_value + majorityelementrange + ( np.random.normal(0,1) ) * curparamseq[4][0]
				newseq[i][0] = perturbed_hdl_value
				newseq[i][1] = perturbed_ldl_value
				newseq[i][2] = perturbed_trigs_value
				newseq[i][3] = perturbed_hba1c_value
				newseq[i][4] = perturbed_ubp_value
			labels[j+totalSize] = 1
			generate_quadratic[j+totalSize] = newseq

		totalSize = totalSize + 75
		
		lossarray,accarray = trainRNNClassifier(generate_quadratic, labels, 5);
		
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
		slope, intercept, r_value, p_value, std_err = stats.linregress(index,generate_quadratic[2800,:20,0])

		linear = []	
		for i in range(20):
			linear.append(slope * index[i] + intercept)
			
			
		#plt.plot(index, hdllist, index, linear)
		plt.plot(index, generate_quadratic[2800,:20,0], index, linear)
		plt.xticks(index)
		plt.ylabel('HDL levels')
		plt.xlabel('NO of visit')
		plt.axis([1, 20, 50, 100])
		plt.show()'''
		
		plt.axis([0, 15, 0, 1])
		plt.scatter(plotcounter, avgacc)
		plt.pause(0.05)
		plotcounter = plotcounter + 1
	#plt.show()
	
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
