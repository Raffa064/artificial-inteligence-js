import NeuralNetwork from './artificial-inteligence.js'

var nn = new NeuralNetwork(1, 5, 5, 2, 1)
nn.crossingOver(nn)
nn.mutate(0.15)