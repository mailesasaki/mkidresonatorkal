"""
Neural net for resonator identification + tuning. Operates in a sliding window fashion
on freqency x attenuation sweep data set. Every point in freq x atten space gets a score 
in one of four classes: (good res, saturated res, underpowered res, no res). Local maxima
in "good res" class are flagged and added to output frequency list. 

The code here (neural net class) only generates the ouput (scored) image; flagging maxima 
and generating freq list are done in findResonatorsWPS.py.

"""

N_CLASSES = 4 #good, saturated, underpowered, bad/no res
COLLISION_FREQ_RANGE = 200.e3
MAX_IMAGES = 10000

ACC_INTERVAL = 500
SAVER_INTERVAL = 10000

import numpy as np
import os
import matplotlib.pyplot as plt
import tensorflow as tf
import mkidresonatorkal.tools as mlt
import mkidcore.sweepdata as sd
import shutil

class WPSNeuralNet(object):
    
    def __init__(self, mlDict):
        self.mlDict = mlDict
        self.nClasses = N_CLASSES
        self.nColors = 2
        if mlDict['useIQV']:
            self.nColors += 1
        if mlDict['useVectIQV']:
            self.nColors += 2

        if not(os.path.isdir(mlDict['modelDir'])):
            os.mkdir(mlDict['modelDir'])
 
        self.trainFile = os.path.join(mlDict['modelDir'], '..', mlDict['trainNPZ'])
        self.imageShape = (mlDict['attenWinBelow'] + mlDict['attenWinAbove'] + 1, mlDict['freqWinSize'], self.nColors)

    def makeTrainData(self):
        trainImages = np.empty((0,) + self.imageShape)
        testImages = np.empty((0,) + self.imageShape)
        trainLabels = np.empty((0, self.nClasses))
        testLabels = np.empty((0, self.nClasses))
        if 'trainCenterIQV' in self.mlDict:
            centerIQV = self.mlDict['trainCenterIQV']
        else:
            centerIQV = False
        if 'trainRandomFreqOffs' in self.mlDict:
            trainRandomFreqOffs = self.mlDict['trainRandomFreqOffs']
        else:
            trainRandomFreqOffs = True

        for i, rawTrainFile in enumerate(self.mlDict['rawTrainFiles']):
            rawTrainFile = os.path.join(self.mlDict['trainFileDir'], rawTrainFile)
            rawTrainMDFile = os.path.join(self.mlDict['trainFileDir'], self.mlDict['rawTrainLabels'][i])
            trainMD = sd.SweepMetadata(file=rawTrainMDFile)
            trainSweep = sd.FreqSweep(rawTrainFile)

            goodResMask = ~np.isnan(trainMD.atten)
            attenblock = np.tile(trainSweep.atten, (len(goodResMask),1))
            optAttenInds = np.argmin(np.abs(attenblock.T - trainMD.atten), axis=0)
            
           
            goodResMask = goodResMask & ~(optAttenInds < self.mlDict['attenWinBelow'])
            goodResMask = goodResMask & ~(optAttenInds >= (len(trainSweep.atten) - self.mlDict['attenWinAbove']))
            #if self.mlDict['filterMaxedAttens']:
            #    maxAttenInd = np.argmax(trainSweep.atten)
            #    goodResMask = goodResMask & ~(optAttenInds==maxAttenInd)
            #    print 'Filtered', np.sum(rawTrainData.opt_iAttens==maxAttenInd), 'maxed out attens.'

            images = np.zeros((self.mlDict['nImagesPerRes']*self.nClasses*np.sum(goodResMask),) + self.imageShape)
            labels = np.zeros((self.mlDict['nImagesPerRes']*self.nClasses*np.sum(goodResMask), self.nClasses))

            optAttens = trainMD.atten[goodResMask]
            optFreqs = trainMD.freq[goodResMask]
            optAttenInds = optAttenInds[goodResMask]
            
            offResFreqs = np.linspace(np.min(optFreqs), np.max(optFreqs), 10000)
            offResFreqMask = np.ones(len(offResFreqs), dtype=bool)

            for f in optFreqs:
                offResFreqMask &= (np.abs(offResFreqs - f) > COLLISION_FREQ_RANGE)

            offResFreqs = offResFreqs[offResFreqMask]

            imgCtr = 0
            for i in range(np.sum(goodResMask)):
                satResMask = np.ones(len(trainSweep.atten), dtype=bool)
                satResMask[optAttenInds[i] - self.mlDict['trainSatThresh']:] = 0
                satResAttens = trainSweep.atten[satResMask]

                upResMask = np.ones(len(trainSweep.atten), dtype=bool)
                upResMask[:optAttenInds[i] + self.mlDict['trainUPThresh']] = 0
                upResMask[-self.mlDict['attenWinAbove']:] = 0
                upResAttens = trainSweep.atten[upResMask]

                for j in range(self.mlDict['nImagesPerRes']):
                    images[imgCtr] = mlt.makeWPSImageList(trainSweep, optFreqs[i], optAttens[i], self.imageShape[1], 
                        self.imageShape[0], self.mlDict['useIQV'], self.mlDict['useVectIQV'], 
                        normalizeBeforeCenter=self.mlDict['normalizeBeforeCenter'], centerIQV=False, randomFreqOffs=False)[0][0] #good image
                    labels[imgCtr] = np.array([1, 0, 0, 0])
                    imgCtr += 1
                    if np.any(satResMask):
                        try:
                            satResAtten = satResAttens[-1] #go down in atten from lowest-power sat res
                            satResAttens = np.delete(satResAttens, -1) #pick without replacement
                        except IndexError:
                            satResAtten = np.random.choice(trainSweep.atten[satResMask]) #pick a random one if out of attens
                        #freqOffs = (-100.e3)*optFreqs[i]/4.e9*np.random.random() #sat resonators move left, so correct this
                        freqOffs = self.mlDict['trainSatFreqOffs']*optFreqs[i]/4.e9 #(-75.e3)*optFreqs[i]/4.e9
                        if self.mlDict['randomizeTrainSatFreqOffs']:
                            freqOffs *= np.random.random()
                        images[imgCtr] = mlt.makeWPSImageList(trainSweep, optFreqs[i]+freqOffs, satResAtten, self.imageShape[1], 
                            self.imageShape[0], self.mlDict['useIQV'], self.mlDict['useVectIQV'],
                            normalizeBeforeCenter=self.mlDict['normalizeBeforeCenter'], centerIQV=centerIQV, randomFreqOffs=trainRandomFreqOffs)[0][0] #saturated image
                        labels[imgCtr] = np.array([0, 1, 0, 0])
                        imgCtr += 1

                    if np.any(upResMask):
                        try:
                            upResAtten = upResAttens[0] #go up in atten from highest-power UP res
                            upResAttens = np.delete(upResAttens, 0) #pick without replacement
                        except IndexError:
                            upResAtten = np.random.choice(trainSweep.atten[upResMask])
                        images[imgCtr] = mlt.makeWPSImageList(trainSweep, optFreqs[i], upResAtten, self.imageShape[1], 
                            self.imageShape[0], self.mlDict['useIQV'], self.mlDict['useVectIQV'],
                            normalizeBeforeCenter=self.mlDict['normalizeBeforeCenter'], centerIQV=False, randomFreqOffs=trainRandomFreqOffs)[0][0] #upurated image
                        labels[imgCtr] = np.array([0, 0, 1, 0])
                        imgCtr += 1

                    offResF = np.random.choice(offResFreqs)
                    offResAtt = np.random.choice(trainSweep.atten)
                    images[imgCtr] = mlt.makeWPSImageList(trainSweep, offResF, offResAtt, self.imageShape[1], 
                        self.imageShape[0], self.mlDict['useIQV'], self.mlDict['useVectIQV'],
                        normalizeBeforeCenter=self.mlDict['normalizeBeforeCenter'], centerIQV=False)[0][0] #off resonance image
                    labels[imgCtr] = np.array([0, 0, 0, 1])
                    imgCtr += 1

            images = images[:imgCtr]
            labels = labels[:imgCtr]

            trainImages = np.append(trainImages, images, axis=0)
            trainLabels = np.append(trainLabels, labels, axis=0)

            print('File ', rawTrainFile, ': Added ', str(len(labels)), ' training images from ' , str(np.sum(goodResMask)), ' resonators')


        allInds = np.arange(len(trainImages), dtype=np.int)
        testInds = np.random.choice(allInds, int(len(allInds)*(1-self.mlDict['trainFrac'])), replace=False)
        trainInds = np.setdiff1d(allInds, testInds)

        testImages = trainImages[testInds]
        testLabels = trainLabels[testInds]
        trainImages = trainImages[trainInds]
        trainLabels = trainLabels[trainInds]

        print('Saving', len(trainImages), 'train images and', len(testImages), 'test images')
        
        np.savez(self.trainFile, trainImages=trainImages, trainLabels=trainLabels,
                testImages=testImages, testLabels=testLabels)

    def initializeAndTrainModel(self, debug=False, saveGraph=False):
        if not os.path.isfile(self.trainFile):
            print('Could not find train file. Making new training images from initialFile')
            self.makeTrainData()

        print('Loading images from ', self.trainFile)
        trainData = np.load(self.trainFile)
        trainImages = trainData['trainImages']
        trainLabels = trainData['trainLabels']
        testImages = trainData['testImages']
        testLabels = trainData['testLabels']

        if self.mlDict['overfitTest']:
            trainImages = trainImages[:30]
            trainLabels = trainLabels[:30]
            testLabels = trainLabels
            testImages = trainImages

        if self.mlDict['centerDataset']:
            self.meanTrainImage = np.mean(trainImages, axis=0)
            trainImages = trainImages - self.meanTrainImage
            testImages = testImages - self.meanTrainImage
            print('Subtracting mean image:', self.meanTrainImage)
        else:
            self.meanTrainImage = np.zeros(trainImages[0].shape)

        
        print('Number of training images:', np.shape(trainImages), ' Number of test images:', np.shape(testImages))

        for k, v in self.mlDict.items():
            tf.compat.v1.add_to_collection('mlDict', tf.constant(value=v, name=k))

        tf.compat.v1.add_to_collection('meanTrainImage', tf.constant(value=self.meanTrainImage, name='image'))
       
        model = tf.keras.models.Sequential([
            tf.keras.layers.Conv2D(filters=self.mlDict['num_filt1'], 
                                   kernel_size=self.mlDict['conv_win1'],
                                   activation=self.mlDict['activation'],
                                   input_shape=(7, 30, 3),
                                   padding=self.mlDict['padding']),
            tf.keras.layers.MaxPooling2D(pool_size=self.mlDict['n_pool1']), 
            tf.keras.layers.Conv2D(filters=self.mlDict['num_filt2'], 
                                   kernel_size=self.mlDict['conv_win2'],
                                   activation=self.mlDict['activation'], 
                                   padding=self.mlDict['padding']),
            tf.keras.layers.MaxPooling2D(pool_size=self.mlDict['n_pool2']),
            tf.keras.layers.Conv2D(filters=self.mlDict['num_filt3'],
                                   kernel_size=self.mlDict['conv_win3'],
                                   activation=self.mlDict['activation'],
                                   padding=self.mlDict['padding']),
            tf.keras.layers.MaxPooling2D(pool_size=self.mlDict['n_pool3']),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(units=self.mlDict['first_neuron_layer']),
            tf.keras.layers.Dropout(rate=self.mlDict['keep_prob']),
            tf.keras.layers.Dense(units=self.mlDict['second_neuron_layer']),
            tf.keras.layers.Softmax()
            ])

        adam_optimizer= tf.keras.optimizers.Adam(learning_rate=self.mlDict['learning_rate'])

        model.compile(optimizer=adam_optimizer, 
                      loss=tf.keras.losses.CategoricalCrossentropy(),
                      metrics=['accuracy']
                      )
        
        fitmodel = model.fit(
            trainImages, trainLabels,
            validation_data = (testImages, testLabels),
            epochs=self.mlDict['trainEpochs']
            )
        
        modelhyperparams = fitmodel.history
        loss = modelhyperparams['loss']
        accuracy = modelhyperparams['accuracy']
        val_loss = modelhyperparams['val_loss']
        val_accuracy = modelhyperparams['val_accuracy']
        
        print('Accuracy of model in testing: ', val_accuracy, '%')
        print('Loss of model in testing: ', val_loss, '%')
        
        Epochs = range(1, len(loss)+1)
        
        plt.plot(Epochs, loss, 'b', label="Training Loss")
        plt.plot(Epochs, val_loss, 'r', label="Validation Loss")
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig('/mnt/c/Users/autum/OneDrive/Desktop/Tensorflow Project/Tensorflow Models/training_loss8.png')
        
        plt.close()

        plt.plot(Epochs, accuracy, 'b', label="Training Accuracy")
        plt.plot(Epochs, val_accuracy, 'r', label="Validation Accuracy")
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.savefig('/mnt/c/Users/autum/OneDrive/Desktop/Tensorflow Project/Tensorflow Models/training_accuracy8.png')

        model.save('/mnt/c/Users/autum/OneDrive/Desktop/Tensorflow Project/Tensorflow Models/saved_model8')

        return shutil.copyfile('/mnt/c/Users/autum/mkidresonatorkal/mkidresonatorkal/mlDictremovedheaders.cfg', '/mnt/c/Users/autum/OneDrive/Desktop/Tensorflow Project/Tensorflow Models/saved_model8/mlDict_new.cfg')        
        
        tf.compat.v1.reset_default_graph()
        
        self.sess.close()

            
                

