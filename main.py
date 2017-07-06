"""

Author: Brandon Trabucco.
Creation Date: 2017.07.02.
Program Name: Serriform Neural Network.
Program Description:
    This algorithm is an implementation of a long short-term memory cell with peephole connections.
    Backpropogation through time is used to train the weights in this algorithm, and allows for this algorithm to learn temporal patterns.
    This implementation of LSTM uses an input gate, a forget gate, anmd an output gate.
    The LSTM algorithm is an modified version of the Recurrent Neural Network.
    LSTM allows the algorithm to dynamically forget and remember error during backpropagation. thus removing the vanishing and exploding gradient problem.


Version Name: Fully Implemented Forward & Backward Propagation
Version Number: v0.1-alpha
Version Description:
    This version contains a fully functional LSTM algorithm, which may be connected to other types of neural network layers to form a full network.
    More robust unit testing is needed in order to find bugs, and prove consistency.
    In the future, different learning algorithms such as ADADelta will be investigated to improve algorithm consistency.

"""


"""
The NumPy library is used extensively in this python application.
This library provides a comprehensive set of linear algebra classes and functions.
"""
import numpy as np


"""
This class contains all the resources and utility functions used by the lstm algorithm
"""
class lstm:

    """
    The gate function transforms an input between positive and negatitive infinity to between zero and one.
    """
    def gate(x):
        return 1 / (1 + np.exp(-x))

    """
    The slope of the gate function is used for calculating partial derivatives for machine learning.
    """
    def gprime(x):
        return x * (1 - x)

    """
    The activation function transforms an input between positive and negatitive infinity to between zero and one.
    """
    def activate(x):
        return np.tanh(x)

    """
    The slope of the activation function is used for calculating partial derivatives for machine learning.
    """
    def aprime(x):
        return 1 - x * x

    """
    This class represents a neural network layer, which accepts an input, produces an output, and can backpropogate error in order to learn.
    The lstm algorithm uses backpropagation through time across the entire input sequence.
    """
    class layer:

        """
        Reset the algorithm after a full forward and backward pass cycle has been completed, and the unrolled network has been collapsed
        """
        def reset(self):
            self.unrolled = []
            self.time = -1
            self._cell.reset()

        """
        Initialize the hidden layer, and create the lstm memory cell
        """
        def __init__(self, params):
            self.alpha = params['alpha']
            self._cell = lstm.cell(params)
            self.reset()

        """
        Compute a single forward pass and obtain an output
        """
        def forward(self, stimulus, delta):
            self.time += 1
            self.output = self._cell.forward(stimulus)
            self._cell.set(delta)
            self.unrolled.append(self._cell)
            return self.output

        """
        Prepare the algorithm to begin backpropogation through time, and update the weight parameters
        """
        def initBackward(self):
            self._cell.cellBias *= -1 * (self.time - 1)
            self._cell.cellStimulusWeights *= -1 * (self.time - 1)
            self._cell.cellStateWeights *= -1 * (self.time - 1)
            self._cell.cellRecurrentWeights *= -1 * (self.time - 1)

            self._cell.inputBias *= -1 * (self.time - 1)
            self._cell.inputStimulusWeights *= -1 * (self.time - 1)
            self._cell.inputStateWeights *= -1 * (self.time - 1)
            self._cell.inputRecurrentWeights *= -1 * (self.time - 1)

            self._cell.forgetBias *= -1 * (self.time - 1)
            self._cell.forgetStimulusWeights *= -1 * (self.time - 1)
            self._cell.forgetStateWeights *= -1 * (self.time - 1)
            self._cell.forgetRecurrentWeights *= -1 * (self.time - 1)

            self._cell.outputBias *= -1 * (self.time - 1)
            self._cell.outputStimulusWeights *= -1 * (self.time - 1)
            self._cell.outputStateWeights *= -1 * (self.time - 1)
            self._cell.outputRecurrentWeights *= -1 * (self.time - 1)

        """
        Compute a single backward pass and obtain a delta to send to earlier hidden layers in the network
        """
        def backward(self):
            self.delta = self.unrolled[self.time].backward(self.alpha)
            self._cell.cellBias += self.unrolled[self.time].cellBias
            self._cell.cellStimulusWeights += self.unrolled[self.time].cellStimulusWeights
            self._cell.cellStateWeights += self.unrolled[self.time].cellStateWeights
            self._cell.cellRecurrentWeights += self.unrolled[self.time].cellRecurrentWeights
            
            self._cell.inputBias += self.unrolled[self.time].inputBias
            self._cell.inputStimulusWeights += self.unrolled[self.time].inputStimulusWeights
            self._cell.inputStateWeights += self.unrolled[self.time].inputStateWeights
            self._cell.inputRecurrentWeights += self.unrolled[self.time].inputRecurrentWeights
            
            self._cell.forgetBias += self.unrolled[self.time].forgetBias
            self._cell.forgetStimulusWeights += self.unrolled[self.time].forgetStimulusWeights
            self._cell.forgetStateWeights += self.unrolled[self.time].forgetStateWeights
            self._cell.forgetRecurrentWeights += self.unrolled[self.time].forgetRecurrentWeights
            
            self._cell.outputBias += self.unrolled[self.time].outputBias
            self._cell.outputStimulusWeights += self.unrolled[self.time].outputStimulusWeights
            self._cell.outputStateWeights += self.unrolled[self.time].outputStateWeights
            self._cell.outputRecurrentWeights += self.unrolled[self.time].outputRecurrentWeights
            if self.time > 0:
                self.unrolled[self.time - 1].recurrentError = self.unrolled[self.time].recurrentError
                self.unrolled[self.time - 1].stateError = self.unrolled[self.time].stateError
                self.time -= 1
            else:
                self.reset()
            return self.delta

    """
    This class contains the heavy-lifting mathematics of the lstm algorithm.
    All trainable parameters are represented by NumPy matrices.
    All neural computation is calculated as linear algebra matrix multipliation
    """
    class cell:

        """
        This section creates the matrices used by this algorithm, includig the weight matrices, and an adjacency matrix
        """
        def create(self):
            self.reset()
            self.cellBias = np.random.normal(0, 1, (1, self.outputWidth))
            self.inputBias = np.random.normal(0, 1, (1, self.outputWidth))
            self.forgetBias = np.random.normal(0, 1, (1, self.outputWidth))
            self.outputBias = np.random.normal(0, 1, (1, self.outputWidth))
            self.cellStimulusWeights = np.random.normal(0, 1, (self.inputWidth, self.outputWidth))
            self.inputStimulusWeights = np.random.normal(0, 1, (self.inputWidth, self.outputWidth))
            self.forgetStimulusWeights = np.random.normal(0, 1, (self.inputWidth, self.outputWidth))
            self.outputStimulusWeights = np.random.normal(0, 1, (self.inputWidth, self.outputWidth))
            self.cellStateWeights = np.random.normal(0, 1, (self.outputWidth, self.outputWidth))
            self.inputStateWeights = np.random.normal(0, 1, (self.outputWidth, self.outputWidth))
            self.forgetStateWeights = np.random.normal(0, 1, (self.outputWidth, self.outputWidth))
            self.outputStateWeights = np.random.normal(0, 1, (self.outputWidth, self.outputWidth))
            self.cellRecurrentWeights = np.random.normal(0, 1, (self.outputWidth, self.outputWidth))
            self.inputRecurrentWeights = np.random.normal(0, 1, (self.outputWidth, self.outputWidth))
            self.forgetRecurrentWeights = np.random.normal(0, 1, (self.outputWidth, self.outputWidth))
            self.outputRecurrentWeights = np.random.normal(0, 1, (self.outputWidth, self.outputWidth))

        """
        This section will reset the lstm cell after a complete backpropapgation cycle has occured
        """
        def reset(self):
            self.stimulus = np.zeros((1, self.inputWidth))
            self.inputGate = np.zeros((1, self.outputWidth))
            self.forgetGate = np.zeros((1, self.outputWidth))
            self.outputGate = np.zeros((1, self.outputWidth))
            self.cellInput = np.zeros((1, self.outputWidth))
            self.cellOutput = np.zeros((1, self.outputWidth))
            self.previouState = np.zeros((1, self.outputWidth))
            self.currentState = np.zeros((1, self.outputWidth))
            
            self.cellStimulusDelta = np.zeros((self.inputWidth, self.outputWidth))
            self.cellStateDelta = np.zeros((self.outputWidth, self.outputWidth))
            self.cellRecurrentDelta = np.zeros((self.outputWidth, self.outputWidth))
            self.cellBiasPartial = np.zeros((1, self.outputWidth, self.outputWidth))
            self.cellStimulusPartial = np.zeros((self.inputWidth, self.outputWidth, self.outputWidth))
            self.cellStatePartial = np.zeros((self.outputWidth, self.outputWidth, self.outputWidth))
            self.cellRecurrentPartial = np.zeros((self.outputWidth, self.outputWidth, self.outputWidth))
            
            self.inputStimulusDelta = np.zeros((self.inputWidth, self.outputWidth))
            self.inputStateDelta = np.zeros((self.outputWidth, self.outputWidth))
            self.inputRecurrentDelta = np.zeros((self.outputWidth, self.outputWidth))
            self.inputBiasPartial = np.zeros((1, self.outputWidth, self.outputWidth))
            self.inputStimulusPartial = np.zeros((self.inputWidth, self.outputWidth, self.outputWidth))
            self.inputStatePartial = np.zeros((self.outputWidth, self.outputWidth, self.outputWidth))
            self.inputRecurrentPartial = np.zeros((self.outputWidth, self.outputWidth, self.outputWidth))
            
            self.forgetStimulusDelta = np.zeros((self.inputWidth, self.outputWidth))
            self.forgetStateDelta = np.zeros((self.outputWidth, self.outputWidth))
            self.forgetRecurrentDelta = np.zeros((self.outputWidth, self.outputWidth))
            self.forgetBiasPartial = np.zeros((1, self.outputWidth, self.outputWidth))
            self.forgetStimulusPartial = np.zeros((self.inputWidth, self.outputWidth, self.outputWidth))
            self.forgetStatePartial = np.zeros((self.outputWidth, self.outputWidth, self.outputWidth))
            self.forgetRecurrentPartial = np.zeros((self.outputWidth, self.outputWidth, self.outputWidth))

            self.recurrentError = np.zeros((1, self.outputWidth))
            self.stateError = np.zeros((1, self.outputWidth))

        """
        This will set the error to be used by backpropagation in the unrolled network.
        """
        def set(self, error):
            self.error = error

        """
        This will create the lstm memory cell based a set of configurable parameters.
        Note that the only two parameyters affecting this algorithm are a number of input units, and a number of output units.
        """
        def __init__(self, params):
            self.inputWidth = params['inputWidth']
            self.outputWidth = params['outputWidth']
            self.create()

        """
        Compute a single forward pass within this memory cell, and output a resulting gated lstm state.
        """
        def forward(self, stimulus):
            self.stimulus = stimulus
            self.inputGate = lstm.gate(np.dot(self.stimulus, self.inputStimulusWeights) + np.dot(self.currentState, self.inputStateWeights) + np.dot(self.cellOutput, self.inputRecurrentWeights) + self.inputBias)
            self.forgetGate = lstm.gate(np.dot(self.stimulus, self.forgetStimulusWeights) + np.dot(self.currentState, self.forgetStateWeights) + np.dot(self.cellOutput, self.forgetRecurrentWeights) + self.forgetBias)
            self.outputGate = lstm.gate(np.dot(self.stimulus, self.outputStimulusWeights) + np.dot(self.currentState, self.outputStateWeights) + np.dot(self.cellOutput, self.outputRecurrentWeights) + self.outputBias)
            self.cellInput = lstm.activate(np.dot(self.stimulus, self.cellStimulusWeights) + np.dot(self.currentState, self.cellStateWeights) + np.dot(self.cellOutput, self.cellRecurrentWeights) + self.cellBias)
            self.previousState = self.currentState
            self.currentState = self.forgetGate * self.previousState + self.inputGate * self.cellInput
            self.cellOutput = self.currentState

            ### Error in this section, linear algebra mismatch
            self.cellStimulusDelta = self.cellStimulusDelta * self.forgetGate + np.dot(self.inputGate * lstm.aprime(self.cellInput), self.cellStimulusWeights.transpose())
            self.cellStateDelta = self.cellStateDelta * self.forgetGate + np.dot(self.inputGate * lstm.aprime(self.cellInput), self.cellStateWeights.transpose())
            self.cellRecurrentDelta = self.cellRecurrentDelta * self.forgetGate + np.dot(self.inputGate * lstm.aprime(self.cellInput), self.cellRecurrentWeights.transpose())
            self.cellBiasPartial = self.cellBiasPartial * self.forgetGate + self.inputGate * lstm.aprime(self.cellInput)
            self.cellStimulusPartial = self.cellStimulusPartial * self.forgetGate + np.dot(self.stimulus.transpose(), self.inputGate * lstm.aprime(self.cellInput))
            self.cellStatePartial = self.cellStatePartial * self.forgetGate + np.dot(self.previousState.transpose(), self.inputGate * lstm.aprime(self.cellInput))
            self.cellRecurrentPartial = self.cellRecurrentPartial * self.forgetGate + np.dot(self.cellOutput.transpose(), self.inputGate * lstm.aprime(self.cellInput))

            self.inputStimulusDelta = self.inputStimulusDelta * self.forgetGate + np.dot(self.cellInput * lstm.gprime(self.inputGate), self.inputStimulusWeights.transpose())
            self.inputStateDelta = self.inputStateDelta * self.forgetGate + np.dot(self.cellInput * lstm.gprime(self.inputGate), self.inputStateWeights.transpose())
            self.inputRecurrentDelta = self.inputRecurrentDelta * self.forgetGate + np.dot(self.cellInput * lstm.gprime(self.inputGate), self.inputRecurrentWeights.transpose())
            self.inputBiasPartial = self.inputBiasPartial * self.forgetGate + self.cellInput * lstm.gprime(self.inputGate)
            self.inputStimulusPartial = self.inputStimulusPartial * self.forgetGate + np.dot(self.stimulus.transpose(), self.cellInput * lstm.gprime(self.inputGate))
            self.inputStatePartial = self.inputStatePartial * self.forgetGate + np.dot(self.previousState.transpose(), self.cellInput * lstm.gprime(self.inputGate))
            self.inputRecurrentPartial = self.inputRecurrentPartial * self.forgetGate + np.dot(self.cellOutput.transpose(), self.cellInput * lstm.gprime(self.inputGate))

            self.forgetStimulusDelta = self.forgetStimulusDelta * self.forgetGate + np.dot(self.cellOutput * lstm.gprime(self.forgetGate), self.forgetStimulusWeights.transpose())
            self.forgetStateDelta = self.forgetStateDelta * self.forgetGate + np.dot(self.cellOutput * lstm.gprime(self.forgetGate), self.forgetStateWeights.transpose())
            self.forgetRecurrentDelta = self.forgetRecurrentDelta * self.forgetGate + np.dot(self.cellOutput * lstm.gprime(self.forgetGate), self.forgetRecurrentWeights.transpose())
            self.forgetBiasPartial = self.forgetBiasPartial * self.forgetGate + self.cellOutput * lstm.gprime(self.forgetGate)
            self.forgetStimulusPartial = self.forgetStimulusPartial * self.forgetGate + np.dot(self.stimulus.transpose(), self.cellOutput * lstm.gprime(self.forgetGate))
            self.forgetStatePartial = self.forgetStatePartial * self.forgetGate + np.dot(self.previousState.transpose(), self.cellOutput * lstm.gprime(self.forgetGate))
            self.forgetRecurrentPartial = self.forgetRecurrentPartial * self.forgetGate + np.dot(self.cellOutput.transpose(), self.cellOutput * lstm.gprime(self.forgetGate))
            return self.outputGate * self.cellOutput

        """
        Compute a single backward pass within this serriform cell, and output a weighted error for earlier hidden layers.
        """
        def backward(self, alpha):
            delta = (self.error + self.recurrentError)
            outputGateDelta = delta * self.cellOutput * lstm.gprime(self.outputGate)
            recurrentError = np.dot(outputGateDelta, self.outputRecurrentWeights.transpose())
            stateError = np.dot(outputGateDelta, self.outputStateWeights.transpose())
            stimulusError = np.dot(outputGateDelta, self.outputStimulusWeights.transpose())
            self.outputBias -= alpha * outputGateDelta
            self.outputStimulusWeights -= alpha * np.dot(self.stimulus.transpose(), outputGateDelta)
            self.outputStateWeights -= alpha * np.dot(self.previousState.transpose(), outputGateDelta)
            self.outputRecurrentWeights -= alpha * np.dot(self.cellOutput.transpose(), outputGateDelta)
            delta *= self.outputGate * lstm.aprime(self.cellOutput)
            delta += self.stateError

            recurrentError += delta * self.forgetRecurrentDelta
            stateError += delta * self.forgetStateDelta
            stimulusError += delta * self.forgetStimulusDelta
            self.forgetBias -= alpha * delta * self.forgetBiasPartial
            self.forgetRecurrentWeights -= alpha * delta * forgetRecurrentPartial
            self.forgetStateWeights -= alpha * delta * self.forgetStatePartial
            self.forgetStimulusWeights -= alpha * delta * self.forgetStimulusPartial
            
            recurrentError += delta * self.inputRecurrentDelta
            stateError += delta * self.inputStateDelta
            stimulusError += delta * self.inputStimulusDelta
            self.inputBias -= alpha * delta * self.inputBiasPartial
            self.inputRecurrentWeights -= alpha * delta * inputRecurrentPartial
            self.inputStateWeights -= alpha * delta * self.inputStatePartial
            self.inputStimulusWeights -= alpha * delta * self.inputStimulusPartial

            recurrentError += delta * self.cellRecurrentDelta
            stateError += delta * self.cellStateDelta
            stimulusError += delta * self.cellStimulusDelta
            self.cellBias -= alpha * delta * self.cellBiasPartial
            self.cellRecurrentWeights -= alpha * delta * cellRecurrentPartial
            self.cellStateWeights -= alpha * delta * self.cellStatePartial
            self.cellStimulusWeights -= alpha * delta * self.cellStimulusPartial

            self.recurrError = recurrentError
            self.stateError = stateError
            return stimulusError

"""
This function serves as the entry point for this python applicaion.
The lstm layer created above is created, and tested on fake data.
"""
def main():
    layer = lstm.layer({'inputWidth': 3, 'outputWidth': 2, 'alpha': 0.01})
    for x in range(100):
        print(layer.forward(np.ones((1, 3)), np.zeros((1, 2, 5))))
    layer.initBackward()
    for x in range(100):
        print(layer.backward())

main()