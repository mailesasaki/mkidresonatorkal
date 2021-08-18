import wpsnnmkidkal2
from mkidcore.config import ConfigThing

mlDict = ConfigThing()
mlDict_file = 'mlDict.yaml'
mlDict.load(mlDict_file)

trial1 = wpsnnmkidkal2.WPSNeuralNet(mlDict)
trial1.initializeAndTrainModel()
