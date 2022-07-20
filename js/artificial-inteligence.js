import log from './logger.js'

const randomGaussian = () => -1 + Math.random() * 2

export default class NeuralNetwork {
    static version = '0.0.2'
    info
    hiddenLayers
    outputLayer
    
    constructor(inputCount, hiddenLayersCount, hiddenLayerNeuronCount, outputLayerNeuronCount) {
        if (inputCount.instance) { //when use an "info" object as param
            hiddenLayersCount = inputCount.hiddenLayersCount
            hiddenLayerNeuronCount = inputCount.hiddenLayerNeuronCount
            outputLayerNeuronCount = inputCount.outputLayerNeuronCount
            inputCount = inputCount.inputCount
        }

        this.info = {
            inputCount: inputCount,
            hiddenLayersCount: hiddenLayersCount,
            hiddenLayerNeuronCount: hiddenLayerNeuronCount,
            outputLayerNeuronCount: outputLayerNeuronCount,
            instance: this
        }
        log('NEURAL NETWORK INFO:<br>inputCount: '+inputCount+'<br>hiddenLayersCount: '+hiddenLayersCount+'<br>hiddenLayerNeuronCount: '+hiddenLayerNeuronCount+'<br>outputLayerNeuronCount: '+outputLayerNeuronCount)
    
        log('Creating HIDDEN LAYERS...', 'warning')
        this.hiddenLayers = new Array(hiddenLayersCount)
        for (let i = 0; i < hiddenLayersCount; i++) {
            this.hiddenLayers[i] = new Array(hiddenLayerNeuronCount)
            for (let j = 0; j < hiddenLayerNeuronCount; j++) {
                this.hiddenLayers[i][j] = new Neuron(inputCount)
            }
            inputCount = hiddenLayerNeuronCount
        }
        
        log('Creating OUTPUT LAYER...', 'warning')
        this.outputLayer = new Array(outputLayerNeuronCount)
        for (let i = 0; i < outputLayerNeuronCount; i++) {
            this.outputLayer[i] = new Neuron(inputCount)
        }
    }

    predict(input) {
        for (const l in this.hiddenLayers) {
            var currentLayerOutput = []
            for (const n in this.hiddenLayers[l]) {
                currentLayerOutput.push(this.hiddenLayers[l][n].predict(input))
            }
            input = currentLayerOutput
        }
        var result = []
        for (const n in this.outputLayer) {
            result.push(this.outputLayer[n].predict(input))
        }
        return result
    }

    //This method mix the neurons of all layers of two networks generating an new agent
    crossingOver(other) {
        let child = new NeuralNetwork(this.info)
        
        var splitSize, splitStart, dbg
        dbg = '<div style="text-align: center;"><strong>CROSSING OVERING:</strong><br>'

        //Mixing hidden layers
        for (let i = 0; i < this.info.hiddenLayersCount; i++) {
            splitSize = Math.floor(Math.random() * this.info.hiddenLayerNeuronCount)
            splitStart = Math.floor(Math.random() * (this.info.hiddenLayerNeuronCount - splitSize))
            for (let j = 0; j < this.info.hiddenLayerNeuronCount; j++) {
                child.hiddenLayers[i][j] = [this.hiddenLayers[i][j], other.hiddenLayers[i][j]][0+(j >= splitStart && j < splitStart + splitSize)]
                dbg += ['<span style="color: #f00">A</span>', '<span style="color: #08f">B</span>'][0+(j >= splitStart && j < splitStart + splitSize)]
            }
            dbg += '<br>'
        }
        
        //Mixing output layer
        splitSize = Math.floor(Math.random() * this.info.outputLayerNeuronCount)
        splitStart = Math.floor(Math.random() * (this.info.outputLayerNeuronCount - splitSize))
        for (let i = 0; i < this.info.outputLayerNeuronCount; i++) {
            child.outputLayer[i] = [this.outputLayer[i], other.outputLayer[i]][0+(i >= splitStart && i < splitStart + splitSize)]
            dbg += ['<span style="color: #fa0">A</span>', '<span style="color: #0f8">B</span>'][0+(i >= splitStart && i < splitStart + splitSize)]
        }

        log(dbg+'</div>')

        return child
    }

    //Apply mutation in all weights and bias depending by the mutation rate
    mutate(rate) {
        var random, dbg = '<div style="text-align: center;"><strong>MUTATING:</strong><br>'

        //Mutating hidden layers
        for (let i = 0; i < this.info.hiddenLayersCount; i++) {
            for (let j = 0; j < this.info.hiddenLayerNeuronCount; j++) {
                random = Math.random()
                dbg += ['<span style="color: #ccc">K</span>', '<span style="color: #0f8">C</span>'][0+(random < rate)]
                if (random < rate) {
                    random = Math.random()
                    const neuron = this.hiddenLayers[i][j]
                    if (random < rate) {
                        neuron.bias += randomGaussian()
                    }

                    const weights = neuron.weights
                    for (let k = 0; k < weights.length; k++) {
                        if (random < rate) {
                            weights[k] += randomGaussian()
                        }
                    }
                }
            }
            dbg += '<br>'
        }
        
        //Mutating output layer
        for (let i = 0; i < this.info.outputLayerNeuronCount; i++) {
            random = Math.random()
            dbg += ['<span style="color: #ccc">K</span>', '<span style="color: #f80">C</span>'][0+(random < rate)]
            if (random < rate) {
                random = Math.random()
                const neuron = this.outputLayer[i]
                if (random < rate) {
                    neuron.bias += randomGaussian()
                }

                const weights = neuron.weights
                for (let j = 0; j < weights.length; j++) {
                    if (random < rate) {
                        weights[j] += randomGaussian()
                    }
                }
            }
        }
        log(dbg+'</div>')
    }
}

export class Neuron {
    bias 
    weights

    constructor(inputCount) {
        try {
            this.weights = new Array(inputCount+1)
            for (let i = 0; i < inputCount+1; i++) {
                this.weights[i] = randomGaussian()
            }
            this.bias = randomGaussian()
            log('A new neuron has created', 'sucess')
        } catch(error) {
            log('Error on create a neuron: '+error, 'error')
        }
    }
    
    predict(input) {
        var u = this.bias * this.weights[0]
        for (let i = 1; i < this.weights.length; i++) {
            u += this.weights[i] * input[i-1]
        }
        var result = Math.tanh(u)
        log('PREDICT:<br>* input = '+JSON.stringify(input)+'<br>* weights = '+JSON.stringify(this.weights)+'<br>* bias = '+JSON.stringify(this.bias)+'<br>* u = '+u+'<br>* f(u) = '+result)
        return result
    }
}

export class NetworTrainer {
    createAgent
    updateAgent
    getFitnessOf
    activeAgents = []
    inactiveAgents = []

    update() {
        for (a in this.activeAgents) {
            const agent = this.activeAgent[a]
            this.updateAgent(agent)
        }
        
    }
}