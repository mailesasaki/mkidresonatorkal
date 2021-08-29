import wpsnnmkidkal2
import mkidcore.config as config
import argparse

parser = argparse.ArgumentParser(description='ML model trainer')
parser.add_argument('mlDict', help='yaml file containing mlDict information')
args=parser.parse_args()

if os.path.isfile(args.mlDict) == False:
        raise Exception('mlDict file does not exist')
    
mlDict_file = args.mlDict
mlDict = config.load(mlDict_file)

trial1 = wpsnnmkidkal2.WPSNeuralNet(mlDict)
trial1.initializeAndTrainModel()
