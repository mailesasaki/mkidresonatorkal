import numpy as np
import random
import tensorflow as tf
import logging
import os, sys, glob
from mkidresonatorkal.python3readdict import ReadDict


def makeWPSImageList(freqSweep, centerFreqList, centerAtten, nFreqs, nAttens, useIQV, useVectIQV, centerIQV=False, normalizeBeforeCenter=False, randomFreqOffs=False):
    centerFreqList = np.atleast_1d(centerFreqList) #allow number too
    winCenters = freqSweep.freqs[:, freqSweep.nlostep//2]
    toneInds = np.argmin(np.abs(centerFreqList - np.tile(winCenters, (len(centerFreqList), 1)).T), axis=0)
    uniqueToneInds, fullToneInds = np.unique(toneInds, return_inverse=True)
    toneFreqList = freqSweep.freqs[uniqueToneInds, :]
    attens = freqSweep.atten
    assert np.all(toneFreqList[fullToneInds, 0] < centerFreqList) and np.all(centerFreqList < toneFreqList[fullToneInds, -1]), 'centerFreq(s) out of range'

    iValList = freqSweep.i[:, uniqueToneInds, :]
    qValList = freqSweep.q[:, uniqueToneInds, :]
    iqVelList = freqSweep.iqvel[:, uniqueToneInds, :]

    centerFreqInds = np.argmin(np.abs(centerFreqList - toneFreqList[fullToneInds].T), axis=0)
    centerAttenInd = np.argmin(np.abs(freqSweep.atten - centerAtten))

    if centerIQV: #slow, do not use for inference
        startFreqInds = centerFreqInds - int(np.floor(nFreqs/2.))
        endFreqInds = centerFreqInds + int(np.ceil(nFreqs/2.))
        for i in range(len(centerFreqInds)):
            centerFreqInds[i] = np.argmax(iqVelList[centerAttenInd, fullToneInds[i], 
                max(0, startFreqInds[i]):endFreqInds[i]]) + max(0, startFreqInds[i])

    if randomFreqOffs:
        centerFreqInds += int(nFreqs*np.random.random(centerFreqInds.shape) - nFreqs/2)

    startFreqInds = centerFreqInds - int(np.floor(nFreqs/2.))
    endFreqInds = centerFreqInds + int(np.ceil(nFreqs/2.))

    if np.min(startFreqInds) < 0:
        startFreqPads = -np.min(startFreqInds)
        startFreqInds += startFreqPads #pad all tones w/ startFreqPads, so shift starting index
        endFreqInds += startFreqPads
    else:
        startFreqPads = 0

    if np.max(endFreqInds) > iqVelList.shape[2] + startFreqPads:
        endFreqPads = np.max(endFreqInds) - iqVelList.shape[2]
    else:
        endFreqPads = 0

    assert freqSweep.atten[0] <= centerAtten <= freqSweep.atten[-1], 'centerAtten out of range'
    startAttenInd = centerAttenInd - int(np.floor(nAttens/2.))
    endAttenInd = centerAttenInd + int(np.ceil(nAttens/2.))

    if startAttenInd < 0:
        startAttenPads = -startAttenInd
        startAttenInd += startAttenPads
        endAttenInd += startAttenPads
    else:
        startAttenPads = 0

    if endAttenInd > len(freqSweep.atten) + startAttenPads:
        endAttenPads = endAttenInd - len(freqSweep.atten)
    else:
        endAttenPads = 0

    if startFreqPads > 0 or endFreqPads > 0 or startAttenPads > 0 or endAttenPads > 0:
        toneFreqList = np.pad(toneFreqList, ((startAttenPads, endAttenPads), (startFreqPads, max(0, endFreqPads-1))), 'edge')
        iValList = np.pad(iValList, ((startAttenPads, endAttenPads), (0, 0), (startFreqPads, max(0, endFreqPads-1))), 'edge')
        qValList = np.pad(qValList, ((startAttenPads, endAttenPads), (0, 0), (startFreqPads, max(0, endFreqPads-1))), 'edge')
        iqVelList = np.pad(iqVelList, ((startAttenPads, endAttenPads), (0, 0), (startFreqPads, endFreqPads)), 'edge')
        attens = np.pad(attens, (startAttenPads, endAttenPads), 'edge')

    #at this point, we have freq lists, Is, Qs, IQVels w/ necessary padding, indexed by (atten, tone, freq). 
    #Need to select windows and normalize - we have freqInds and attenInds to do this for each window

    freqSlices = np.array(list(map(range, startFreqInds, endFreqInds)))
    attenSlices = np.tile(list(range(startAttenInd, endAttenInd)), (len(centerFreqList), nFreqs, 1))

    iValList = np.transpose(iValList, (1, 0, 2)) #shape is now (nTone, nAtten, nFreq)
    qValList = np.transpose(qValList, (1, 0, 2))
    iqVelList = np.transpose(iqVelList, (1, 0, 2))
    iValList = iValList[(fullToneInds, attenSlices.T, freqSlices.T)].transpose(2, 0, 1) #hope this works....
    qValList = qValList[(fullToneInds, attenSlices.T, freqSlices.T)].transpose(2, 0, 1) #now has shape (nTone, nAtten, nFreq)
    iqVelList = iqVelList[(fullToneInds, attenSlices.T, freqSlices.T)].transpose(2, 0, 1) 
    toneFreqList = toneFreqList[(fullToneInds, freqSlices.T)].T
    attens = attens[startAttenInd:endAttenInd]

    if normalizeBeforeCenter:
        res_mag = np.sqrt(np.mean(iValList**2 + qValList**2, axis=2))
        iValList = iValList.transpose((2, 0, 1))/res_mag #shape is (nFreq, nTone, nAtten)
        qValList = qValList.transpose((2, 0, 1))/res_mag
        iqVelList = iqVelList.transpose((2, 0, 1))/res_mag
        iValList = iValList - np.mean(iValList, axis=0)
        qValList = qValList - np.mean(qValList, axis=0)
        iqVelList = iqVelList - np.mean(iqVelList, axis=0)

        iValList = iValList.transpose(1, 2, 0) #reshape to (nTone, nAtten, nFreq)
        qValList = qValList.transpose(1, 2, 0)
        iqVelList = iqVelList.transpose(1, 2, 0)

    else:
        raise Exception('You are making a mistake....')

    if useVectIQV:
        raise Exception('Not yet implemented')

    if useIQV:
        images = np.stack((iValList, qValList, iqVelList), axis=3)
    else:
        images = np.stack((iValList, qValList), axis=3)

    return images, attens, toneFreqList

def makeWPSImage(freqSweep, centerFreq, centerAtten, nFreqs, nAttens, useIQV, useVectIQV, centerIQV=False, normalizeBeforeCenter=False):
    """
    dataObj: FreqSweep object
    """
    winCenters = freqSweep.freqs[:, freqSweep.nlostep//2]
    toneInd = np.argmin(np.abs(centerFreq - winCenters)) #index of resonator tone to use
    toneFreqs = freqSweep.freqs[toneInd, :]
    assert toneFreqs[0] < centerFreq < toneFreqs[-1], 'centerFreq out of range'

    centerFreqInd = np.argmin(np.abs(toneFreqs - centerFreq))
    startFreqInd = centerFreqInd - int(np.floor(nFreqs/2.))
    endFreqInd = centerFreqInd + int(np.ceil(nFreqs/2.))

    if startFreqInd < 0:
        startFreqPads = 0 - startFreqInd
        startFreqInd = 0
    else:
        startFreqPads = 0
    if endFreqInd > len(toneFreqs):
        endFreqPads = endFreqInd - len(toneFreqs)
        endFreqInd = len(toneFreqs)
    else:
        endFreqPads = 0

    assert freqSweep.atten[0] <= centerAtten <= freqSweep.atten[-1], 'centerAtten out of range'
    centerAttenInd = np.argmin(np.abs(freqSweep.atten - centerAtten))
    startAttenInd = centerAttenInd - int(np.floor(nAttens/2.))
    endAttenInd = centerAttenInd + int(np.ceil(nAttens/2.))

    if startAttenInd < 0:
        startAttenPads = 0 - startAttenInd
        startAttenInd = 0
    else:
        startAttenPads = 0
    if endAttenInd > len(freqSweep.atten):
        endAttenPads = endAttenInd - len(freqSweep.atten)
        endAttenInd = len(freqSweep.atten)
    else:
        endAttenPads = 0

    #SELECT
    freqs = toneFreqs[startFreqInd:endFreqInd]
    attens = freqSweep.atten[startAttenInd:endAttenInd]
    iVals = freqSweep.i[startAttenInd:endAttenInd, toneInd, startFreqInd:endFreqInd]
    qVals = freqSweep.q[startAttenInd:endAttenInd, toneInd, startFreqInd:endFreqInd]

    if centerIQV:
        iqVels = np.sqrt(np.diff(iVals, axis=1)**2 + np.diff(qVals, axis=1)**2)
        raise Exception('Not implemented')


    #NORMALIZE
    if normalizeBeforeCenter:
        res_mag = np.sqrt(np.mean(iVals**2 + qVals**2, axis=1))
        iVals = np.transpose(np.transpose(iVals)/res_mag)
        qVals = np.transpose(np.transpose(qVals)/res_mag)
        iVals = np.transpose(np.transpose(iVals) - np.mean(iVals, axis=1))
        qVals = np.transpose(np.transpose(qVals) - np.mean(qVals, axis=1))

    else:
        iVals = np.transpose(np.transpose(iVals) - np.mean(iVals, axis=1))
        qVals = np.transpose(np.transpose(qVals) - np.mean(qVals, axis=1))
        res_mag = np.sqrt(np.mean(iVals**2 + qVals**2, axis=1))
        iVals = np.transpose(np.transpose(iVals)/res_mag)
        qVals = np.transpose(np.transpose(qVals)/res_mag)

    iqVels = np.sqrt(np.diff(iVals, axis=1)**2 + np.diff(qVals, axis=1)**2)
    iVels = np.diff(iVals, axis=1)
    qVels = np.diff(qVals, axis=1)

    iqVels = np.transpose(np.transpose(iqVels) - np.mean(iqVels, axis=1))
    #iqVels = np.transpose(np.transpose(iqVels)/res_mag)
    #iqVels /= np.sqrt(np.mean(iqVels**2))

    iVels = np.transpose(np.transpose(iVels) - np.mean(iVels, axis=1))
    #iVels = np.transpose(np.transpose(iVels)/res_mag)
    #iVels /= np.sqrt(np.mean(iVels**2))

    qVels = np.transpose(np.transpose(qVels) - np.mean(qVels, axis=1))
    #qVels = np.transpose(np.transpose(qVels)/res_mag)
    #qVels /= np.sqrt(np.mean(qVels**2))

    #PAD
    iVals = np.pad(iVals, ((startAttenPads, endAttenPads), (startFreqPads, endFreqPads)), 'edge')
    qVals = np.pad(qVals, ((startAttenPads, endAttenPads), (startFreqPads, endFreqPads)), 'edge')
    iqVels = np.pad(iqVels, ((startAttenPads, endAttenPads), (startFreqPads, endFreqPads+1)), 'edge')
    iVels = np.pad(iVels, ((startAttenPads, endAttenPads), (startFreqPads, endFreqPads+1)), 'edge')
    qVels = np.pad(qVels, ((startAttenPads, endAttenPads), (startFreqPads, endFreqPads+1)), 'edge')
    freqs = np.pad(freqs, (startFreqPads, endFreqPads), 'edge')
    attens = np.pad(attens, (startAttenPads, endAttenPads), 'edge')

    image = np.dstack((iVals, qVals))
    if useIQV:
        image = np.dstack((image, iqVels))
    if useVectIQV:
        image = np.dstack((image, iVels))
        image = np.dstack((image, qVels))

    return image, attens, freqs


def get_ml_model(modelDir=''):
    new_model = tf.keras.models.load_model(modelDir)

    mlDictFile = modelDir + '/mlDict_new.cfg'
    mlDict = ReadDict()
    mlDict.readFromFile(mlDictFile)

    if 'normalizeBeforeCenter' not in mlDict: #maintain backwards compatibility with old models
        print('Adding key: normalizeBeforeCenter')
        mlDict['normalizeBeforeCenter'] = False

    return mlDict, new_model


def next_batch(trainImages, trainLabels, batch_size):
    '''selects a random batch of batch_size from trainImages and trainLabels'''
    perm = random.sample(list(range(len(trainImages))), batch_size)
    trainImagesBatch = trainImages[perm]
    trainLabelsBatch = trainLabels[perm]
    #print 'next_batch trImshape', np.shape(trainImages)
    return trainImagesBatch, trainLabelsBatch


