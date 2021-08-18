import wpsnnmkidkal2
import mkidcore.config

mlDict = mkidcore.config.load(<'mlDict.yaml'>)

trial1 = wpsnnmkidkal2.WPSNeuralNet(mlDict)
trial1.initializeAndTrainModel()
