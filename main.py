
# coding: utf-8

# In[24]:


import PIL
from PIL import Image
from PIL import ImageOps
import numpy as np
#from glob import glob
import os
import idx2numpy
import math
import matplotlib.pyplot as plt


# In[2]:


class MNIST:
    ##Import MNIST dataset. Produces 28*28 matrix for each image. 60000 data points.
    ##Normalizes dataset so that all pixel values are between 0 and 1
    def __init__(self):
        f_read = open('MNIST_images/train-images-idx3-ubyte', 'rb') 
        self.train_input = idx2numpy.convert_from_file(f_read)
        self.train_input = self.reshapeAddInterceptScaleDown(self.train_input,255.)
        f_read = open('MNIST_images/train-labels-idx1-ubyte', 'rb')
        self.train_labels = idx2numpy.convert_from_file(f_read)
        f_read = open('MNIST_images/t10k-images-idx3-ubyte', 'rb') 
        self.test_input = idx2numpy.convert_from_file(f_read)
        self.test_input = self.reshapeAddInterceptScaleDown(self.test_input,255.)
        f_read = open('MNIST_images/t10k-labels-idx1-ubyte', 'rb')
        self.test_labels = idx2numpy.convert_from_file(f_read)

    ##Reshape the array for each image to be a vector. 
    ##Rescale vectors to be smaller for easier calculation.
    ##Add intercept (column of 1s)
    def reshapeAddInterceptScaleDown(self,inputData,scale):
        input_flattened = np.reshape(inputData,(inputData.shape[0], -1))
        input_flattened_scaled = input_flattened/scale
        input_intercept_added = np.insert(input_flattened_scaled, 0, 1, axis=1)
        return input_intercept_added


# In[31]:


#Import the USPS dataset from folders the specified directory
class USPS:
    def __init__(self):
        trainImagesDir = "proj3_images/Numerals/"
        input_pixels_array = []#np.empty((0, 100))
        labels_array = []
        basewidth = 28
        i = 0
        while i<10:
            folder = trainImagesDir + str(i) + "/"
            for filename in os.listdir(folder):
                if filename.endswith(".png"):
                    filepath = folder + filename
                    pixVal = self.reshape_flip(filepath)
                    input_pixels_array.append(pixVal)
                    labels_array.append(i)
            i = i+1
        self.inputArr = np.array(input_pixels_array)/255.
        self.labels = np.array(labels_array)
        self.labelsOneHot = self.oneHotStack()
    
    #Open the image file, reshape the image, invert the pixel values.
    def reshape_flip(self,filepath):
        try:
            im = Image.open(filepath, 'r')
            im = PIL.ImageOps.grayscale(im)
        except:
            print "could not open",filepath      
        old_size = im.size  # old_size[0] is in (width, height) format
        ratio = float(28)/max(old_size)
        new_size = tuple([int(x*ratio) for x in old_size])
        # use thumbnail() or resize() method to resize the input image
        # thumbnail is a in-place operation
        # im.thumbnail(new_size, Image.ANTIALIAS)
        img = im.resize(new_size, Image.ANTIALIAS)
        # create a new image and paste the resized on it
        new_im = Image.new('L', (28, 28),255)
        new_im.paste(img, ((28-new_size[0])//2,
                            (28-new_size[1])//2))
        inverted_image = PIL.ImageOps.invert(new_im)
        pix_val = [1] + list(inverted_image.getdata())
        return pix_val
    
    def oneHotStack(self):
        a = np.zeros([self.labels.size,10])
        i=0
        while i<self.labels.size:
            a[i,self.labels[i]] = 1
            i=i+1
        return a


# # Logistic Regression (with minibatch)

# In[11]:


class Weights:
    #Create a new weights array of size numClasses x numFeatures
    #Each row of the weights array is the weights vector for a particular class
    def __init__(self,numClasses,numFeatures):
        self.k = numClasses
        self.weights = np.ones([numClasses,numFeatures])*0.01

    #Calculate the probability that one or more images (an input vector of pixel values) belongs to each class
    #Returns a vector of class probabilities, or a matrix in the case of multiple images.
    def classProbs(self,inputData):
        expActivations = np.exp(np.matmul(inputData,self.weights.T))
        sumExpActivations = np.sum(expActivations,axis=1)
        return expActivations / sumExpActivations[:,None]
        
    #Creates a one hot vector the length of the number of classes based on the class index provided
    def oneHotStack(self,classLabels):
        a = np.zeros([classLabels.size,self.k])
        i=0
        while i<classLabels.size:
            a[i,classLabels[i]] = 1
            i=i+1
        return a
    
    #Calculates the cross entropy error for a number of images
    def crossEntropyError(self,inputData,labels):
        classProbs = self.classProbs(inputData)
        logClassProbs = np.log(classProbs)
        return -1 * np.sum(np.multiply(logClassProbs,self.oneHotStack(labels))) 
    
    #Used for gradient checking (checkWeightsGrad())
    def ceeOtherWeights(self,inputData,labels,weights):
        expActivations = np.exp(np.matmul(inputData,weights.T))
        sumExpActivations = np.sum(expActivations,axis=1)
        classProbs = expActivations / sumExpActivations[:,None]
        logClassProbs = np.log(classProbs)
        return -1 * np.sum(np.multiply(logClassProbs,self.oneHotStack(labels)))
    
    #Calculates the gradient of the weights vector for any number of images
    def weightsGradBatch(self,inputData,labels):
        return np.matmul((self.classProbs(inputData) - self.oneHotStack(labels)).T,inputData)
    
    #Ensures that the weightsGradBatch calculation is correct. Input data must be for a single datapoint
    def checkWeightsGrad(self,inputData,labels,epsilon):
        wgb = self.weightsGradBatch(inputData,labels)
        #loop over weights, change each weight by +/- epsilon to get manual grad, compare to calculated grad
        manualGradmatrix = np.zeros(wgb.shape)
        for i in range (self.weights.shape[0]):
            for j in range (self.weights.shape[1]):
                wplus = np.array(self.weights, copy=True)
                wplus[i,j] +=epsilon
                wminus = np.array(self.weights, copy=True)
                wminus[i,j] -=epsilon
                cee1 = self.ceeOtherWeights(inputData,labels,wplus)
                cee2 = self.ceeOtherWeights(inputData,labels,wminus)
                manualgrad = (cee1 - cee2)/(2*epsilon)
                #print i,j,"actualgrad",wgb[i,j],"manualgrad",manualgrad,classActivations1,classActivations2
                manualGradmatrix[i,j] = manualgrad
        return manualGradmatrix,wgb
    
    #Perform batch gradient descent for a certain number of epochs. Processes batchSize images at once.
    def batchGD(self,learningRate,inputData,outputData,testInput,testOutput,num_epochs,batchSize):
        N = inputData.shape[0]
        CEETrack = []
        EpochTrack = []
        testaccuracy = []
        trainaccuracy = []
        trainacc = self.evaluation(inputData,outputData)
        testacc = self.evaluation(testInput,testOutput)
        CEE = self.crossEntropyError(inputData,outputData)
        print "INITIALLY CEE:",CEE,"WeightsNorm:",np.linalg.norm(self.weights),"TrainAcc:",trainacc,"TestAcc:",testacc
        for epoch in range(num_epochs):
            #print "Epoch",epoch
            for i in range(N/batchSize):
                lower_bound= i*batchSize 
                upper_bound= min((i+1)*batchSize,  N)
                batch = inputData[lower_bound:upper_bound,:]
                outbatch = outputData[lower_bound:upper_bound]
                wg = self.weightsGradBatch(batch,outbatch)
                self.weights = self.weights - learningRate*wg
            trainacc = self.evaluation(inputData,outputData)
            trainaccuracy.append(trainacc)
            testacc = self.evaluation(testInput,testOutput)
            testaccuracy.append(testacc)
            #CEE = self.crossEntropyError(inputData,outputData)
            #CEETrack.append(CEE)
            EpochTrack.append(epoch)
            #print "CEE:",CEE,"WeightsNorm:",np.linalg.norm(self.weights),"TrainAcc:",trainacc,"TestAcc:",testacc
        plt.plot(EpochTrack, testaccuracy, 'r-',linewidth=0.5)
        plt.plot(EpochTrack, trainaccuracy, 'b-',linewidth=0.5)
        plt.legend(('Test accuracy', 'Train accuracy'))
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.show()
        return CEETrack,EpochTrack,testaccuracy,trainaccuracy

    #Evaluates a set of evaluation inputs and outputs using the current weights to determine the % correct predictions.
    def evaluation(self,evaluation_input,evaluation_output):
        probs = self.classProbs(evaluation_input)
        predict = np.argmax(probs,axis=1)
        correct = 0
        for n in range (evaluation_output.size):
            if np.argmax(probs[n,:]) == evaluation_output[n]:
                correct = correct+1
        return float(correct)/float(evaluation_output.size)


# In[32]:


#Import MNIST data
mnistD = MNIST()
#Import USPS data
usps = USPS()


# In[6]:


#Check weights grad
mnistW = Weights(10,mnistD.train_input.shape[1])
mnistW.checkWeightsGrad(mnistD.train_input[10:11,:],mnistD.train_labels[10:11],0.00001)


# In[19]:


#Generate starting weights, run batch gradient descent, evaluate on test data
lrs = [0.001,0.0001,0.00001]
batchsizes = [10,30,50,100,1000,10000]
for l in lrs:
    for b in batchsizes:
        mnistW = Weights(10,mnistD.train_input.shape[1])
        mnistW.batchGD(l,mnistD.train_input,mnistD.train_labels,mnistD.test_input,mnistD.test_labels,500,b)
        CEEtrain = mnistW.crossEntropyError(mnistD.train_input,mnistD.train_labels)
        CEEtest = mnistW.crossEntropyError(mnistD.test_input,mnistD.test_labels)
        trainacc = mnistW.evaluation(mnistD.train_input,mnistD.train_labels)
        testacc = mnistW.evaluation(mnistD.test_input,mnistD.test_labels)
        print "Learning Rate:",l,"Train Accuracy:",trainacc,"Test Accuracy:",testacc,"CEE Train:",CEEtrain,"CEE Test:",CEEtest


# # Single Layer Neural Network (backprop implemented from scratch

# In[26]:


class SingleLayerNeuralNetwork:
    def __init__(self,numClasses,numFeatures,numHiddenUnits):
        np.random.seed(5)
        self.k = numClasses
        self.weightsItoH = np.random.random((numFeatures,numHiddenUnits))
        self.weightsHtoO = np.random.random((numHiddenUnits+1,numClasses))
        
    def sigmoidActivation(self,inputVal):
        return 1./(1.+np.exp(-1.*inputVal))
    
    def sigmoidActivationDerivative(self,inputVal):
        return self.sigmoidActivation(inputVal)*(1. - self.sigmoidActivation(inputVal))
    
    #Creates a one hot vector the length of the number of classes based on the class index provided
    def oneHotStack(self,classLabels):
        a = np.zeros([classLabels.size,self.k])
        i=0
        while i<classLabels.size:
            a[i,classLabels[i]] = 1
            i=i+1
        return a
    
    #Shuffles the data and the output (identically) so that the batches for each epoch are not the same
    def shufflesDataAndOutputEqually(self,inputData,outputData):
        assert len(inputData) == len(outputData)
        p = np.random.permutation(len(inputData))
        return inputData[p], outputData[p]
    
    #Calculate the probability that an image (an input vector of pixel values) belongs to each class
    #Returns a vector of class probabilities
    def classProbs(self,inputData):
        hiddenLayer = self.sigmoidActivation(np.matmul(inputData,self.weightsItoH))
        hiddenLayerWithBias = np.insert(hiddenLayer, 0, 1,axis=1)
        outputActivations = np.matmul(hiddenLayerWithBias,self.weightsHtoO)
        expActivations = np.exp(outputActivations)
        sumExpActivations = np.sum(expActivations,axis=1)
        return expActivations / sumExpActivations[:,None]
    
    #Calculates the cross entropy error for a particular image
    def CrossEntropyError(self,inputVector,label):
        classProbs = self.classProbs(inputVector)
        logClassProbs = np.log(classProbs)
        #print np.multiply(logClassProbs,self.oneHotStack(label)).shape
        return -1. * np.sum(np.multiply(logClassProbs,self.oneHotStack(label)))
    
    #Used for gradient checking (checkWeightsGrad())
    def ceeOtherWeights(self,inputData,labels,weightsItoH,weightsHtoO):
        hidden = self.sigmoidActivation(np.matmul(inputData,weightsItoH))
        hiddenAddExtra = np.insert(hidden, 0, 1,axis=1)
        outputActivations = np.matmul(hiddenAddExtra,weightsHtoO)
        expActivations = np.exp(outputActivations)
        sumExpActivations = np.sum(expActivations,axis=1)
        classProbs = expActivations / sumExpActivations[:,None]
        logClassProbs = np.log(classProbs)
        return -1. * np.sum(np.multiply(logClassProbs,self.oneHotStack(labels)))
    
    #Matrix of y-t, where t is a one hot matrix with the correct label index in each row
    def deltaK(self,outputLabels,inputData):
        return self.classProbs(inputData) - self.oneHotStack(outputLabels)
    
    #ZxK array of weight gradients
    def gradWeightsHtoO(self,deltaK,inputVector):
        hiddenLayer = np.insert(self.calcHiddenLayer(inputVector), 0, 1,axis=1)
        return np.matmul(hiddenLayer.T,deltaK)
    
    #Z length vector
    def deltaJ(self,inputVector,deltaK):
        hiddenLayerDerivative = self.calcHiddenLayerDerivative(inputVector) #single case 1 x Z
        weightsByDeltaKDropBias = np.delete(np.matmul(self.weightsHtoO,deltaK.T),0,axis=0) #single case 
        return np.multiply(hiddenLayerDerivative,weightsByDeltaKDropBias.T)
    
    def gradWeightsItoH(self,deltaJ,inputData):
        #print "deltaJ",deltaJ
        return np.matmul(inputData.T,deltaJ)
    
    #Ensures that the weightsGradBatch calculation is correct. Input data must be for a single datapoint
    def checkWeightsGrad(self,inputData,labels,epsilon):
        #forward propagation
        hiddenLayer = np.insert(self.sigmoidActivation(np.matmul(inputData,self.weightsItoH)), 0, 1,axis=1)
        outputActivations = np.matmul(hiddenLayer,self.weightsHtoO)
        expActivations = np.exp(outputActivations)
        sumExpActivations = np.sum(expActivations,axis=1)
        yHat = expActivations / sumExpActivations[:,None]
        #print yHat,labels
        deltak = yHat - self.oneHotStack(labels)
        gradHtoO = np.matmul(hiddenLayer.T,deltak)/inputData.shape[0]
        hiddenLayerDerivative = self.sigmoidActivationDerivative(np.matmul(inputData,self.weightsItoH))
        weightsByDeltaKDropBias = np.delete(np.matmul(self.weightsHtoO,deltak.T),0,axis=0)
        deltaj = np.multiply(hiddenLayerDerivative,weightsByDeltaKDropBias.T)
        gradItoH = np.matmul(inputData.T,deltaj)/inputData.shape[0]
        #loop over weights, change each weight by +/- epsilon to get manual grad, compare to calculated grad
        gradHtoOManual = np.zeros(gradHtoO.shape)
        for i in range (gradHtoO.shape[0]):
            for j in range (gradHtoO.shape[1]):
                wplus = np.array(self.weightsHtoO, copy=True)
                print wplus[i,j]
                wplus[i,j] +=epsilon
                print wplus[i,j]
                wminus = np.array(self.weightsHtoO, copy=True)
                wminus[i,j] -=epsilon
                #NEED TO FIX THIS BIT - CALC CROSS ENTROPY ERROR WITH DIFF WEIGHTS
                cee1 = self.ceeOtherWeights(inputData,labels,self.weightsItoH,wplus)
                cee2 = self.ceeOtherWeights(inputData,labels,self.weightsItoH,wminus)
                #print cee1-cee2
                manualgrad = (cee1 - cee2)/(2*epsilon)
                print i,j,"actualgrad",gradHtoO[i,j],"manualgrad",manualgrad,cee1,cee2,"actualCEE",self.CrossEntropyError(inputData,labels)
                gradHtoOManual[i,j] = manualgrad
        gradItoHManual = np.zeros(gradItoH.shape)
        for i in range (gradItoH.shape[0]):
            for j in range (gradItoH.shape[1]):
                wplus = np.array(self.weightsItoH, copy=True)
                wplus[i,j] += epsilon
                wminus = np.array(self.weightsItoH, copy=True)
                wminus[i,j] -= epsilon
                cee1 = self.ceeOtherWeights(inputData,labels,wplus,self.weightsHtoO)
                cee2 = self.ceeOtherWeights(inputData,labels,wminus,self.weightsHtoO)
                manualgrad = (cee1 - cee2)/(2*epsilon)
                #print i,j,"actualgrad",wgb[i,j],"manualgrad",manualgrad,classActivations1,classActivations2
                gradItoHManual[i,j] = manualgrad
        return gradHtoO,gradHtoOManual,gradItoH,gradItoHManual
    
    #Evaluates a set of evaluation inputs and outputs using the current weights to determine the % correct predictions.
    def evaluation(self,evaluation_input,evaluation_output):
        probs = self.classProbs(evaluation_input)
        predict = np.argmax(probs,axis=1)
        correct = 0
        for n in range (evaluation_output.size):
            if predict[n] == evaluation_output[n]:
                correct = correct+1
        return float(correct)/float(evaluation_output.size)
        
    def trainNN(self,inputData,outputData,learningRate,batchSize,numEpochs):
        if learningRate == "adaptLR":
            learningRate = 1.0
        N = inputData.shape[0]
        for epoch in range(numEpochs):
            if epoch%100 == 0:
                learningRate = learningRate*0.9
                print "Epoch",epoch
            inputData,outputData = self.shufflesDataAndOutputEqually(inputData,outputData)
            for i in range(N/batchSize):
                lower_bound= i*batchSize 
                upper_bound= min((i+1)*batchSize,N)
                #print lower_bound,upper_bound
                batch = inputData[lower_bound:upper_bound,:]
                outbatch = outputData[lower_bound:upper_bound]
                
                #forward propagation
                hiddenLayer = np.insert(self.sigmoidActivation(np.matmul(batch,self.weightsItoH)), 0, 1,axis=1)
                outputActivations = np.matmul(hiddenLayer,self.weightsHtoO)
                expActivations = np.exp(outputActivations)
                #print expActivations.shape
                sumExpActivations = np.sum(expActivations,axis=1)
                #print sumExpActivations.shape
                ##yHat is Nx10 matrix, each row contains class probs for a data point
                yHat = expActivations / sumExpActivations[:,None]
                #print yHat.shape
                
                #back propagation to get weights grad
                deltak = yHat - self.oneHotStack(outbatch)
                gradHtoO = np.matmul(hiddenLayer.T,deltak)/batchSize
                hiddenLayerDerivative = self.sigmoidActivationDerivative(np.matmul(batch,self.weightsItoH))
                weightsByDeltaKDropBias = np.delete(np.matmul(self.weightsHtoO,deltak.T),0,axis=0)
                deltaj = np.multiply(hiddenLayerDerivative,weightsByDeltaKDropBias.T)
                gradItoH = np.matmul(batch.T,deltaj)/batchSize
                
                #Update weights
                self.weightsItoH -= learningRate*gradItoH
                self.weightsHtoO -= learningRate*gradHtoO
                
            #Print weights norm
            #print np.linalg.norm(self.weightsItoH),"ItoHWeightsNorm"
            #print np.linalg.norm(self.weightsHtoO),"HtoOWeightsNorm"
                
            
            if epoch%100==0:
                print "Evaluating..."
                CEE = self.CrossEntropyError(inputData,outputData)
                evaluation = self.evaluation(inputData,outputData)
                print "Cross Entropy Error:",CEE,"trainRateCorrect",evaluation
        evaluation = self.evaluation(mnistD.test_input,mnistD.test_labels)
        


# In[27]:


#Initialize SNN for MNIST with five hidden units
mnistSNN = SingleLayerNeuralNetwork(10,mnistD.train_input.shape[1],5)


# In[29]:


#Train neural network with 
mnistSNN.trainNN(mnistD.train_input,mnistD.train_labels,"adaptLR",100000,10)


# In[33]:


#Evaluate on USPS dataset
mnistSNN.evaluation(usps.inputArr,usps.labels)

