# -*- coding: utf-8 -*-
"""
Created on Fri Apr  9 17:33:10 2021

@author: autum
"""


import wpsnnmkidkal2
from mkidresonatorkal.python3readdict import ReadDict

mlDictFile = 'mlDictremovedheaders.cfg'
mlDict = ReadDict()
mlDict.readFromFile(mlDictFile)


trial1 = wpsnnmkidkal2.WPSNeuralNet(mlDict)
trial1.initializeAndTrainModel()