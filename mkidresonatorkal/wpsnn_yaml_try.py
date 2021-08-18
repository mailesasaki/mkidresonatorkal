import wpsnnmkidkal2
import mkidcore.config as config

mlDict_file = 'mlDict.yml'
mlDict = config.load(mlDict_file)

trial1 = wpsnnmkidkal2.WPSNeuralNet(mlDict)
trial1.initializeAndTrainModel()
