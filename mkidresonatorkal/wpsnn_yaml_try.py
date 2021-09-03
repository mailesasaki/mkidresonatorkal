"""
Trains the model when given an mlDict file
"""

import wpsnnmkidkal2
import mkidcore.config as config
import argparse
import os

parser = argparse.ArgumentParser(description='ML model trainer')
parser.add_argument('mlDict', help='yaml file containing mlDict information')
parser.add_argument('model_name', help='destination of ML model')
args=parser.parse_args()

if os.path.isfile(args.mlDict) == False:
        raise Exception('mlDict file does not exist')
    
mlDict_file = args.mlDict
model_name = args.model_name
mlDict = config.load(mlDict_file)

trial1 = wpsnnmkidkal2.WPSNeuralNet(mlDict)
trial1.initializeAndTrainModel()
