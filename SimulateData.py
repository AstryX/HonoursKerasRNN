#RNN Classifier for multi-variable synthetic data set classification and prediction for the Honours Project
#Robertas Dereskevicius 2019/03 University of Dundee
import numpy as np
import json
from scipy import stats
import sys
import statistics
import matplotlib.pyplot as plt

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

class DataGenerator:
	def __init__(myObject, fileName):
		with open(fileName) as json_data_file:
		
			data = json.load(json_data_file)
			
			myObject.numFeatures = 5
			
			#Multiple of 3
			myObject.numLinearData = data['numLinearData']
			
			myObject.numLinearDataSingle = int(myObject.numLinearData / 3)
			
			myObject.numExponentialDecreasing = data['numExponentialDecreasing']
			myObject.numExponentialDecreasingFalse = data['numExponentialDecreasingFalse']
			myObject.numExponentialIncreasing = data['numExponentialIncreasing']
			myObject.numExponentialIncreasingFalse = data['numExponentialIncreasingFalse']
			
			myObject.totalDataLen = myObject.numLinearData + 3 * (myObject.numExponentialDecreasing + myObject.numExponentialDecreasingFalse + \
									myObject.numExponentialIncreasing + myObject.numExponentialIncreasingFalse)
			
			myObject.hdl_value = data['hdl_value']
			myObject.ldl_value = data['ldl_value']
			myObject.trigs_value = data['trigs_value']
			myObject.hba1c_value = data['hba1c_value']
			myObject.ubp_value = data['ubp_value']
			
			myObject.hdlconst = data['hdlconst']
			myObject.ldlconst = data['ldlconst']
			myObject.trigsconst = data['trigsconst']
			myObject.hba1cconst = data['hba1cconst']
			myObject.ubpconst = data['ubpconst']
			
			myObject.majorityelementrange = data['majorityelementrange']
			myObject.minorityelementrange = data['minorityelementrange']
			
			myObject.majoritylineardisplacement = data['majoritylineardisplacement']
			myObject.minoritylineardisplacement = data['minoritylineardisplacement']
			
			myObject.expmajoritysteepness = data['expmajoritysteepness']
			myObject.expminoritysteepness = data['expminoritysteepness']
			
			myObject.initialexpshift = data['initialexpshift']	

			myObject.sequencelenstart = data['sequencelenstart']
			myObject.sequencelenend = data['sequencelenend']
			
			myObject.paramconfig = data['paramconfig']

	def getExpHDL(myObject, i, reverse, offsetcoefficient, steepnesParam):
		if reverse == 1:
			return -myObject.expmajoritysteepness*pow(1.2 - steepnesParam, (i+1)) + 70 + myObject.majorityelementrange + (myObject.majorityelementrange*offsetcoefficient)
		else:
			return myObject.expmajoritysteepness*pow(1.2 - steepnesParam, (i+1)) + 70 - (myObject.majorityelementrange*offsetcoefficient)
		

	#Similar boundaries as hdl
		
	def getExpLDL(myObject, i, reverse, offsetcoefficient, steepnesParam):
		if reverse == 1:
			return -myObject.expmajoritysteepness*pow(1.2 - steepnesParam, (i+1)) + 70 + myObject.majorityelementrange + (myObject.majorityelementrange*offsetcoefficient)
		else:
			return myObject.expmajoritysteepness*pow(1.2 - steepnesParam, (i+1)) + 70 - (myObject.majorityelementrange*offsetcoefficient)

	def getExpTRIGS(myObject, i, reverse, offsetcoefficient, steepnesParam):
		if reverse == 1:
			return -myObject.expmajoritysteepness*pow(1.2 - steepnesParam, (i+1)) + 120 + myObject.majorityelementrange + (myObject.majorityelementrange*offsetcoefficient)
		else:
			return myObject.expmajoritysteepness*pow(1.2 - steepnesParam, (i+1)) + 120 - (myObject.majorityelementrange*offsetcoefficient)

	def getExpHBA1C(myObject, i, reverse, offsetcoefficient, steepnesParam):
		if reverse == 1:
			return -myObject.expminoritysteepness*pow(1.2 - steepnesParam, (i+1) - 10) + 3 + myObject.minorityelementrange + (myObject.minorityelementrange*offsetcoefficient)
		else:
			return myObject.expminoritysteepness*pow(1.2 - steepnesParam, (i+1) - 10) + 3 - (myObject.minorityelementrange*offsetcoefficient)

	def getExpUBP(myObject, i, reverse, offsetcoefficient, steepnesParam):
		if reverse == 1:
			return -myObject.expmajoritysteepness*pow(1.2 - steepnesParam, (i+1)) + 95 + myObject.majorityelementrange + (myObject.majorityelementrange*offsetcoefficient)
		else:
			return myObject.expmajoritysteepness*pow(1.2 - steepnesParam, (i+1)) + 95 - (myObject.majorityelementrange*offsetcoefficient)
		
		
	def generateSyntheticData(myObject, drugType, curveType, dataQuantity, currentParameters, isIncreasing = False, isError = False):
		if curveType == 'Linear' and drugType == 'NoDrug':
			#Does it for 3 times
			displacement = 0
			X_generated = np.zeros((int(dataQuantity * len(myObject.majoritylineardisplacement)),20,myObject.numFeatures))	
			y_generated = np.zeros((int(dataQuantity * len(myObject.majoritylineardisplacement))))
			for k in range(len(myObject.majoritylineardisplacement)):
				for j in range(dataQuantity):
					newseq = np.zeros((20,myObject.numFeatures))
					for i in range(20):
						perturbed_hdl_value = myObject.hdl_value + myObject.majoritylineardisplacement[k] + ( np.random.normal(0,1) ) * currentParameters[0][0]
						perturbed_ldl_value = myObject.ldl_value + myObject.majoritylineardisplacement[k] + ( np.random.normal(0,1) ) * currentParameters[1][0]
						perturbed_trigs_value = myObject.trigs_value + myObject.majoritylineardisplacement[k] + ( np.random.normal(0,1) ) * currentParameters[2][0]
						perturbed_hba1c_value = myObject.hba1c_value + myObject.minoritylineardisplacement[k] + ( np.random.normal(0,1) ) * currentParameters[3][0]
						perturbed_ubp_value = myObject.ubp_value + myObject.majoritylineardisplacement[k] + ( np.random.normal(0,1) ) * currentParameters[4][0]
						newseq[i][0] = perturbed_hdl_value
						newseq[i][1] = perturbed_ldl_value
						newseq[i][2] = perturbed_trigs_value
						newseq[i][3] = perturbed_hba1c_value
						newseq[i][4] = perturbed_ubp_value
					y_generated[j+displacement] = 0
					X_generated[j+displacement] = newseq
					
				displacement = displacement + dataQuantity
				
			return X_generated, y_generated	
			
		elif curveType == 'Exponential':
			X_generated = np.zeros((dataQuantity,20,myObject.numFeatures))	
			y_generated = np.zeros((dataQuantity))
			
			if drugType == 'Drug1Drug2':
				if isError == True:
					positionalMajorityShift = 0
					positionalMinorityShift = 0
					if isIncreasing == True:
						positionalMajorityShift = -(myObject.majorityelementrange)
						positionalMinorityShift = -(myObject.minorityelementrange)
					elif isIncreasing == False:
						positionalMajorityShift = myObject.majorityelementrange
						positionalMinorityShift = myObject.minorityelementrange
					#no response
					for j in range(dataQuantity):
						newseq = np.zeros((20,myObject.numFeatures))
						for i in range(20):
							perturbed_hdl_value = myObject.hdl_value + positionalMajorityShift + ( np.random.normal(0,1) ) * currentParameters[0][0]
							perturbed_ldl_value = myObject.ldl_value + positionalMajorityShift + ( np.random.normal(0,1) ) * currentParameters[1][0]
							perturbed_trigs_value = myObject.trigs_value + positionalMajorityShift + ( np.random.normal(0,1) ) * currentParameters[2][0]
							perturbed_hba1c_value = myObject.hba1c_value + positionalMinorityShift + ( np.random.normal(0,1) ) * currentParameters[3][0]
							perturbed_ubp_value = myObject.ubp_value + positionalMajorityShift + ( np.random.normal(0,1) ) * currentParameters[4][0]
							newseq[i][0] = perturbed_hdl_value
							newseq[i][1] = perturbed_ldl_value
							newseq[i][2] = perturbed_trigs_value
							newseq[i][3] = perturbed_hba1c_value
							newseq[i][4] = perturbed_ubp_value
						y_generated[j] = 3
						X_generated[j] = newseq
				elif isError == False:
					isReverse = -1
					positionalMajorityShift = 0
					positionalMinorityShift = 0
					if isIncreasing == True:
						isReverse = 0
						positionalMajorityShift = -(myObject.initialexpshift * myObject.majorityelementrange)
						positionalMinorityShift = -(myObject.initialexpshift * myObject.minorityelementrange)
					elif isIncreasing == False:
						isReverse = 1
						positionalMajorityShift = (myObject.initialexpshift * myObject.majorityelementrange)
						positionalMinorityShift = (myObject.initialexpshift * myObject.minorityelementrange)
					#Quadratic increasing with response
					for j in range(dataQuantity):
						newseq = np.zeros((20,myObject.numFeatures))
						for i in range(20):
							hdl_value_quad = myObject.getExpHDL(i,isReverse,currentParameters[0][1],currentParameters[0][2])
							ldl_value_quad = myObject.getExpLDL(i,isReverse,currentParameters[1][1],currentParameters[1][2])
							trigs_value_quad = myObject.getExpTRIGS(i,isReverse,currentParameters[2][1],currentParameters[2][2])
							hba1c_value_quad = myObject.getExpHBA1C(i,isReverse,currentParameters[3][1],currentParameters[3][2])
							ubp_value_quad = myObject.getExpUBP(i,isReverse,currentParameters[4][1],currentParameters[4][2])
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
						y_generated[j] = 3
						X_generated[j] = newseq	
						'''index = []
						for i in range(20):
							index.append(i+1)

						#slope, intercept, r_value, p_value, std_err = stats.linregress(index,hdllist)
						slope, intercept, r_value, p_value, std_err = stats.linregress(index,newseq[:20,0])

						linear = []	
						for i in range(20):
							linear.append(slope * index[i] + intercept)
							
							
						#plt.plot(index, hdllist, index, linear)
						plt.plot(index, newseq[:20,0], index, linear)
						plt.xticks(index)
						plt.ylabel('Simulated HDL levels with noise')
						plt.xlabel('NO of visit')
						plt.axis([1, 20, 50, 120])
						plt.show()'''
			
			elif drugType == 'Drug1':
				if isError == True:
					positionalMajorityShift = 0
					positionalMinorityShift = 0
					if isIncreasing == True:
						positionalMajorityShift = -(myObject.majorityelementrange)
						positionalMinorityShift = -(myObject.minorityelementrange)
					elif isIncreasing == False:
						positionalMajorityShift = myObject.majorityelementrange
						positionalMinorityShift = myObject.minorityelementrange
					#no response
					for j in range(dataQuantity):
						newseq = np.zeros((20,myObject.numFeatures))
						for i in range(20):
							perturbed_hdl_value = myObject.hdl_value + positionalMajorityShift + ( np.random.normal(0,1) ) * currentParameters[0][0]
							perturbed_ldl_value = myObject.ldl_value + positionalMajorityShift + ( np.random.normal(0,1) ) * currentParameters[1][0]
							perturbed_trigs_value = myObject.trigs_value + ( np.random.normal(0,1) ) * currentParameters[2][0]
							perturbed_hba1c_value = myObject.hba1c_value + ( np.random.normal(0,1) ) * currentParameters[3][0]
							perturbed_ubp_value = myObject.ubp_value + ( np.random.normal(0,1) ) * currentParameters[4][0]
							newseq[i][0] = perturbed_hdl_value
							newseq[i][1] = perturbed_ldl_value
							newseq[i][2] = perturbed_trigs_value
							newseq[i][3] = perturbed_hba1c_value
							newseq[i][4] = perturbed_ubp_value
						y_generated[j] = 1
						X_generated[j] = newseq
				elif isError == False:
					isReverse = -1
					positionalMajorityShift = 0
					positionalMinorityShift = 0
					if isIncreasing == True:
						isReverse = 0
						positionalMajorityShift = -(myObject.initialexpshift * myObject.majorityelementrange)
						positionalMinorityShift = -(myObject.initialexpshift * myObject.minorityelementrange)
					elif isIncreasing == False:
						isReverse = 1
						positionalMajorityShift = (myObject.initialexpshift * myObject.majorityelementrange)
						positionalMinorityShift = (myObject.initialexpshift * myObject.minorityelementrange)
					#Quadratic increasing with response
					for j in range(dataQuantity):
						newseq = np.zeros((20,myObject.numFeatures))
						for i in range(20):
							hdl_value_quad = myObject.getExpHDL(i,isReverse,currentParameters[0][1],currentParameters[0][2])
							ldl_value_quad = myObject.getExpLDL(i,isReverse,currentParameters[1][1],currentParameters[1][2])
							perturbed_hdl_value = hdl_value_quad + positionalMajorityShift + ( np.random.normal(0,1) ) * currentParameters[0][0]
							perturbed_ldl_value = ldl_value_quad + positionalMajorityShift + ( np.random.normal(0,1) ) * currentParameters[1][0]
							perturbed_trigs_value = myObject.trigs_value + ( np.random.normal(0,1) ) * currentParameters[2][0]
							perturbed_hba1c_value = myObject.hba1c_value + ( np.random.normal(0,1) ) * currentParameters[3][0]
							perturbed_ubp_value = myObject.ubp_value + ( np.random.normal(0,1) ) * currentParameters[4][0]
							newseq[i][0] = perturbed_hdl_value
							newseq[i][1] = perturbed_ldl_value
							newseq[i][2] = perturbed_trigs_value
							newseq[i][3] = perturbed_hba1c_value
							newseq[i][4] = perturbed_ubp_value
						y_generated[j] = 1
						X_generated[j] = newseq				
			elif drugType == 'Drug2':
				if isError == True:
					positionalMajorityShift = 0
					positionalMinorityShift = 0
					if isIncreasing == True:
						positionalMajorityShift = -(myObject.majorityelementrange)
						positionalMinorityShift = -(myObject.minorityelementrange)
					elif isIncreasing == False:
						positionalMajorityShift = myObject.majorityelementrange
						positionalMinorityShift = myObject.minorityelementrange
					#no response
					for j in range(dataQuantity):
						newseq = np.zeros((20,myObject.numFeatures))
						for i in range(20):
							perturbed_hdl_value = myObject.hdl_value + ( np.random.normal(0,1) ) * currentParameters[0][0]
							perturbed_ldl_value = myObject.ldl_value + ( np.random.normal(0,1) ) * currentParameters[1][0]
							perturbed_trigs_value = myObject.trigs_value + positionalMajorityShift + ( np.random.normal(0,1) ) * currentParameters[2][0]
							perturbed_hba1c_value = myObject.hba1c_value + positionalMinorityShift + ( np.random.normal(0,1) ) * currentParameters[3][0]
							perturbed_ubp_value = myObject.ubp_value + positionalMajorityShift + ( np.random.normal(0,1) ) * currentParameters[4][0]
							newseq[i][0] = perturbed_hdl_value
							newseq[i][1] = perturbed_ldl_value
							newseq[i][2] = perturbed_trigs_value
							newseq[i][3] = perturbed_hba1c_value
							newseq[i][4] = perturbed_ubp_value
						y_generated[j] = 2
						X_generated[j] = newseq
				elif isError == False:
					isReverse = -1
					positionalMajorityShift = 0
					positionalMinorityShift = 0
					if isIncreasing == True:
						isReverse = 0
						positionalMajorityShift = -(myObject.initialexpshift * myObject.majorityelementrange)
						positionalMinorityShift = -(myObject.initialexpshift * myObject.minorityelementrange)
					elif isIncreasing == False:
						isReverse = 1
						positionalMajorityShift = (myObject.initialexpshift * myObject.majorityelementrange)
						positionalMinorityShift = (myObject.initialexpshift * myObject.minorityelementrange)
					#Quadratic increasing with response
					for j in range(dataQuantity):
						newseq = np.zeros((20,myObject.numFeatures))
						for i in range(20):
							trigs_value_quad = myObject.getExpTRIGS(i,isReverse,currentParameters[2][1],currentParameters[2][2])
							hba1c_value_quad = myObject.getExpHBA1C(i,isReverse,currentParameters[3][1],currentParameters[3][2])
							ubp_value_quad = myObject.getExpUBP(i,isReverse,currentParameters[4][1],currentParameters[4][2])
							perturbed_hdl_value = myObject.hdl_value + ( np.random.normal(0,1) ) * currentParameters[0][0]
							perturbed_ldl_value = myObject.ldl_value + ( np.random.normal(0,1) ) * currentParameters[1][0]
							perturbed_trigs_value = trigs_value_quad + positionalMajorityShift + ( np.random.normal(0,1) ) * currentParameters[2][0]
							perturbed_hba1c_value = hba1c_value_quad + positionalMinorityShift + ( np.random.normal(0,1) ) * currentParameters[3][0]
							perturbed_ubp_value = ubp_value_quad + positionalMajorityShift + ( np.random.normal(0,1) ) * currentParameters[4][0]
							newseq[i][0] = perturbed_hdl_value
							newseq[i][1] = perturbed_ldl_value
							newseq[i][2] = perturbed_trigs_value
							newseq[i][3] = perturbed_hba1c_value
							newseq[i][4] = perturbed_ubp_value
						y_generated[j] = 2
						X_generated[j] = newseq			
			return X_generated, y_generated
			
		return None,None
		
	def padSequenceWithZeros(myObject, totalLen, curStep, dataToPad):
		X_padded = []
		for pad in range(totalLen - curStep - 1):
			featurePad = []
			for zerofeature in range(myObject.numFeatures):
				featurePad.append(0)
			X_padded.append(featurePad)
		X_padded.extend(dataToPad)	
		return X_padded
		
	def retrieveConcatenatedDrugData(myObject, curparamseq):
		tempX, tempY = myObject.generateSyntheticData('NoDrug', 'Linear', myObject.numLinearDataSingle, curparamseq)
		syntheticDataset = tempX
		labels = tempY
		
		tempX, tempY = myObject.generateSyntheticData('Drug1', 'Exponential', myObject.numExponentialIncreasing, curparamseq, True, False)
		syntheticDataset = np.concatenate((syntheticDataset,tempX), axis=0)
		labels = np.concatenate((labels,tempY), axis=0)
		
		tempX, tempY = myObject.generateSyntheticData('Drug1', 'Exponential', myObject.numExponentialIncreasingFalse, curparamseq, True, True)
		syntheticDataset = np.concatenate((syntheticDataset,tempX), axis=0)
		labels = np.concatenate((labels,tempY), axis=0)
		
		tempX, tempY = myObject.generateSyntheticData('Drug1', 'Exponential', myObject.numExponentialDecreasing, curparamseq, False, False)
		syntheticDataset = np.concatenate((syntheticDataset,tempX), axis=0)
		labels = np.concatenate((labels,tempY), axis=0)
		
		tempX, tempY = myObject.generateSyntheticData('Drug1', 'Exponential', myObject.numExponentialDecreasingFalse, curparamseq, False, True)
		syntheticDataset = np.concatenate((syntheticDataset,tempX), axis=0)
		labels = np.concatenate((labels,tempY), axis=0)

		tempX, tempY = myObject.generateSyntheticData('Drug2', 'Exponential', myObject.numExponentialIncreasing, curparamseq, True, False)
		syntheticDataset = np.concatenate((syntheticDataset,tempX), axis=0)
		labels = np.concatenate((labels,tempY), axis=0)
		
		tempX, tempY = myObject.generateSyntheticData('Drug2', 'Exponential', myObject.numExponentialIncreasingFalse, curparamseq, True, True)
		syntheticDataset = np.concatenate((syntheticDataset,tempX), axis=0)
		labels = np.concatenate((labels,tempY), axis=0)
		
		tempX, tempY = myObject.generateSyntheticData('Drug2', 'Exponential', myObject.numExponentialDecreasing, curparamseq, False, False)
		syntheticDataset = np.concatenate((syntheticDataset,tempX), axis=0)
		labels = np.concatenate((labels,tempY), axis=0)
		
		tempX, tempY = myObject.generateSyntheticData('Drug2', 'Exponential', myObject.numExponentialDecreasingFalse, curparamseq, False, True)
		syntheticDataset = np.concatenate((syntheticDataset,tempX), axis=0)
		labels = np.concatenate((labels,tempY), axis=0)
			
		tempX, tempY = myObject.generateSyntheticData('Drug1Drug2', 'Exponential', myObject.numExponentialIncreasing, curparamseq, True, False)
		syntheticDataset = np.concatenate((syntheticDataset,tempX), axis=0)
		labels = np.concatenate((labels,tempY), axis=0)
		
		tempX, tempY = myObject.generateSyntheticData('Drug1Drug2', 'Exponential', myObject.numExponentialIncreasingFalse, curparamseq, True, True)
		syntheticDataset = np.concatenate((syntheticDataset,tempX), axis=0)
		labels = np.concatenate((labels,tempY), axis=0)
		
		tempX, tempY = myObject.generateSyntheticData('Drug1Drug2', 'Exponential', myObject.numExponentialDecreasing, curparamseq, False, False)
		syntheticDataset = np.concatenate((syntheticDataset,tempX), axis=0)
		labels = np.concatenate((labels,tempY), axis=0)
		
		tempX, tempZ = myObject.generateSyntheticData('Drug1Drug2', 'Exponential', myObject.numExponentialDecreasingFalse, curparamseq, False, True)
		syntheticDataset = np.concatenate((syntheticDataset,tempX), axis=0)
		labels = np.concatenate((labels,tempZ), axis=0)
		
		return syntheticDataset, labels

	def splitToMiniBatches(myObject, X_set, y_set, miniBatchSize = 32):
		
		finishedBatchesX = []
		finishedBatchesY = []
		
		tempBatchesX = []
		tempBatchesY = []
		
		for i in range(19):
			tempBatchesX.append([])
			tempBatchesY.append([])
			
		for i in range(len(X_set)):
			sizeIndex = len(X_set[i])-1
			tempBatchesX[sizeIndex].append(X_set[i])
			tempBatchesY[sizeIndex].append(y_set[i])
			
			if len(tempBatchesX[sizeIndex]) == miniBatchSize:
				finishedBatchesX.append(tempBatchesX[sizeIndex])
				finishedBatchesY.append(tempBatchesY[sizeIndex])
				
				tempBatchesX[sizeIndex] = []
				tempBatchesY[sizeIndex] = []
		
		return finishedBatchesX, finishedBatchesY
		
	#Used for the regressor problem that needs different labels
	def constructSyntheticDataPredictions(myObject, X_train, shouldPadWithZeros, shouldCreateMiniBatches = False, miniBatchSize = 32):
		X_seqtrain = []
		y_seqtrain = []
		for sequence in X_train:
			for i in range(myObject.sequencelenstart,myObject.sequencelenend):
				temp_x = []
				for j in range(i+1):
					temp_x.append(sequence[j])
				if(shouldPadWithZeros == True):
					X_padded = myObject.padSequenceWithZeros((len(sequence)-1),i,temp_x)
					X_seqtrain.append(X_padded)
					y_seqtrain.append(sequence[i+1])
				else:
					X_seqtrain.append(temp_x)
					y_seqtrain.append(sequence[i+1])
		
		if shouldCreateMiniBatches == True:
			X_seqtrain, y_seqtrain = myObject.splitToMiniBatches(X_seqtrain, y_seqtrain, miniBatchSize)
		
		return X_seqtrain, y_seqtrain

	def createParamVector(myObject):
		paramvector = []

		for it in myObject.paramconfig:
			paramseq = []

			paramseq.append([ myObject.hdlconst * it[0], it[1], it[2] ] )
			paramseq.append([ myObject.ldlconst * it[0], it[1], it[2] ] )
			paramseq.append([ myObject.trigsconst * it[0], it[1], it[2] ] )
			paramseq.append([ myObject.hba1cconst * it[0], it[1], it[2] ] )
			paramseq.append([ myObject.ubpconst * it[0], it[1], it[2] ] )

			paramvector.append(paramseq)
		
		return paramvector
