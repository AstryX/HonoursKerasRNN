#Data simulation class that generates different biomarkers for varying sequences of different classes for both classification and regression problems, for the Honours Project
#Robertas Dereskevicius 2019/03 University of Dundee
import numpy as np
import json
from scipy import stats
import sys
import statistics
import matplotlib.pyplot as plt	

#Stored data simulation functions within a class that can be instantiated with a parameter file name
class DataGenerator:
#Constructor with initial values that are defined in the formulae and manual docs
	def __init__(myObject, fileName):
		with open(fileName) as json_data_file:
		
			data = json.load(json_data_file)
			
			myObject.numFeatures = 5
			
			#Multiple of len of majoritylineardisplacement length
			myObject.numLinearData = data['numLinearData']
			
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
			
			myObject.numLinearDataSingle = int(myObject.numLinearData / len(myObject.majoritylineardisplacement))
			
			myObject.expmajoritysteepness = data['expmajoritysteepness']
			myObject.expminoritysteepness = data['expminoritysteepness']
			
			myObject.initialexpshift = data['initialexpshift']	

			myObject.sequencelenstart = data['sequencelenstart']
			myObject.sequencelenend = data['sequencelenend']
			
			myObject.paramconfig = data['paramconfig']

	#Formula used for calculating HDL exponential curve time steps
	def getExpHDL(myObject, i, reverse, offsetcoefficient, steepnesParam):
		if reverse == 1:
			return -myObject.expmajoritysteepness*pow(1.2 - steepnesParam, (i+1)) + (myObject.hdl_value - (myObject.majorityelementrange / 2)) + myObject.majorityelementrange + (myObject.majorityelementrange*offsetcoefficient)
		else:
			return myObject.expmajoritysteepness*pow(1.2 - steepnesParam, (i+1)) + (myObject.hdl_value - (myObject.majorityelementrange / 2)) - (myObject.majorityelementrange*offsetcoefficient)
		
	#Formula used for calculating LDL exponential curve time steps
	def getExpLDL(myObject, i, reverse, offsetcoefficient, steepnesParam):
		if reverse == 1:
			return -myObject.expmajoritysteepness*pow(1.2 - steepnesParam, (i+1)) + (myObject.ldl_value - (myObject.majorityelementrange / 2)) + myObject.majorityelementrange + (myObject.majorityelementrange*offsetcoefficient)
		else:
			return myObject.expmajoritysteepness*pow(1.2 - steepnesParam, (i+1)) + (myObject.ldl_value - (myObject.majorityelementrange / 2)) - (myObject.majorityelementrange*offsetcoefficient)

	#Formula used for calculating TRIGS exponential curve time steps
	def getExpTRIGS(myObject, i, reverse, offsetcoefficient, steepnesParam):
		if reverse == 1:
			return -myObject.expmajoritysteepness*pow(1.2 - steepnesParam, (i+1)) + (myObject.trigs_value - (myObject.majorityelementrange / 2)) + myObject.majorityelementrange + (myObject.majorityelementrange*offsetcoefficient)
		else:
			return myObject.expmajoritysteepness*pow(1.2 - steepnesParam, (i+1)) + (myObject.trigs_value - (myObject.majorityelementrange / 2)) - (myObject.majorityelementrange*offsetcoefficient)

	#Formula used for calculating hbA1c exponential curve time steps
	def getExpHBA1C(myObject, i, reverse, offsetcoefficient, steepnesParam):
		if reverse == 1:
			return -myObject.expminoritysteepness*pow(1.2 - steepnesParam, (i+1) - 10) + (myObject.hba1c_value - (myObject.minorityelementrange / 2)) + myObject.minorityelementrange + (myObject.minorityelementrange*offsetcoefficient)
		else:
			return myObject.expminoritysteepness*pow(1.2 - steepnesParam, (i+1) - 10) + (myObject.hba1c_value - (myObject.minorityelementrange / 2)) - (myObject.minorityelementrange*offsetcoefficient)

	#Formula used for calculating systolic blood pressure exponential curve time steps
	def getExpUBP(myObject, i, reverse, offsetcoefficient, steepnesParam):
		if reverse == 1:
			return -myObject.expmajoritysteepness*pow(1.2 - steepnesParam, (i+1)) + (myObject.ubp_value - (myObject.majorityelementrange / 2)) + myObject.majorityelementrange + (myObject.majorityelementrange*offsetcoefficient)
		else:
			return myObject.expmajoritysteepness*pow(1.2 - steepnesParam, (i+1)) + (myObject.ubp_value - (myObject.majorityelementrange / 2)) - (myObject.majorityelementrange*offsetcoefficient)
		
	#Primary data simulation function
	def generateSyntheticData(myObject, drugType, curveType, dataQuantity, currentParameters, isIncreasing = False, isError = False):
		if curveType == 'Linear' and drugType == 'NoDrug':
			displacement = 0
			X_generated = np.zeros((int(dataQuantity * len(myObject.majoritylineardisplacement)),20,myObject.numFeatures))	
			y_generated = np.zeros((int(dataQuantity * len(myObject.majoritylineardisplacement))))
			#Data is slightly perturbed for each NoDrug patient to add variety
			for k in range(len(myObject.majoritylineardisplacement)):
				for j in range(dataQuantity):
					newseq = np.zeros((20,myObject.numFeatures))
					for i in range(20):
						#Linear baseline with linear displacement and perturbations
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
			#Both drugs mean that all 5 biomarkers are out of ranges
			if drugType == 'Drug1Drug2':
				#Error in this case means patient has no reponse to drugs
				if isError == True:
					positionalMajorityShift = 0
					positionalMinorityShift = 0
					#Increasing in this case means that the biomarkers should be below the range
					if isIncreasing == True:
						positionalMajorityShift = -(myObject.majorityelementrange)
						positionalMinorityShift = -(myObject.minorityelementrange)
					elif isIncreasing == False:
						positionalMajorityShift = myObject.majorityelementrange
						positionalMinorityShift = myObject.minorityelementrange
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
				#If drug has a response
				elif isError == False:
					isReverse = -1
					positionalMajorityShift = 0
					positionalMinorityShift = 0
					#Defines initial offsets based on the direction the curve is going
					if isIncreasing == True:
						isReverse = 0
						positionalMajorityShift = -(myObject.initialexpshift * myObject.majorityelementrange)
						positionalMinorityShift = -(myObject.initialexpshift * myObject.minorityelementrange)
					elif isIncreasing == False:
						isReverse = 1
						positionalMajorityShift = (myObject.initialexpshift * myObject.majorityelementrange)
						positionalMinorityShift = (myObject.initialexpshift * myObject.minorityelementrange)
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
			#First two biomarkers are out of range
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
			#Last 3 biomarkers are out of range
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
		
	#Function that pads a sequence with zeros until it is of length timesteps - 1. Used for regression to make sequences of equal length.
	def padSequenceWithZeros(myObject, totalLen, curStep, dataToPad):
		X_padded = []
		for pad in range(totalLen - curStep - 1):
			featurePad = []
			for zerofeature in range(myObject.numFeatures):
				featurePad.append(0)
			X_padded.append(featurePad)
		X_padded.extend(dataToPad)	
		return X_padded
		
	#Function that concatenates many results for different classes and data simulation edge-cases
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

	#Function that orders and splits data objects into mini-batches of sequences with same length in a data set of varying lengths
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
		
	#Used for the regressor problem that needs different labels (next time step biomarkers)
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

	#Simple function used to load and create the parameter vector of varying difficulties
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
