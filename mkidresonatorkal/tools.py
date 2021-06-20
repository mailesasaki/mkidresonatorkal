import numpy as np
import random
import tensorflow as tf
import logging
import os, sys, glob

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
        centerFreqInds += int(nFreqs*np.random.random(centerFreqInds.shape) - nFreqs/2.)

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
    attenSlices = np.tile(range(startAttenInd, endAttenInd), (len(centerFreqList), nFreqs, 1))

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
        iValList = iValList.transpose((2, 0, 1))//res_mag #shape is (nFreq, nTone, nAtten)
        qValList = qValList.transpose((2, 0, 1))//res_mag
        iqVelList = iqVelList.transpose((2, 0, 1))//res_mag
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

def get_ml_model(modelDir=''):
    modelList = glob.glob(os.path.join(modelDir, '*.meta'))
    if len(modelList) > 1:
        raise Exception('Multiple models (.meta files) found in directory: ' + modelDir)
    elif len(modelList) == 0:
        raise Exception('No models (.meta files) found in directory ' + modelDir)
    model = modelList[0]
    getLogger(__name__).info('Loading good model from %s', model)
    sess = tf.Session()
    saver = tf.train.import_meta_graph(model)
    saver.restore(sess, tf.train.latest_checkpoint(os.path.dirname(model)))

    graph = tf.get_default_graph()
    x_input = graph.get_tensor_by_name('inputImage:0')
    y_output = graph.get_tensor_by_name('outputLabel:0')
    keep_prob = graph.get_tensor_by_name('keepProb:0')
    is_training = graph.get_tensor_by_name('isTraining:0')

    mlDict = {}
    for param in tf.get_collection('mlDict'):
        mlDict[param.op.name] = param.eval(session=sess)

    if 'normalizeBeforeCenter' not in mlDict: #maintain backwards compatibility with old models
        print('Adding key: normalizeBeforeCenter')
        mlDict['normalizeBeforeCenter'] = False

    return mlDict, sess, graph, x_input, y_output, keep_prob, is_training


def next_batch(trainImages, trainLabels, batch_size):
    '''selects a random batch of batch_size from trainImages and trainLabels'''
    perm = random.sample(list(range(len(trainImages))), batch_size)
    trainImagesBatch = trainImages[perm]
    trainLabelsBatch = trainLabels[perm]
    #print 'next_batch trImshape', np.shape(trainImages)
    return trainImagesBatch, trainLabelsBatch


