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
This class contains all the resources and utility functions used by the lstm algorithm.
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
    This class serves as a utility to create multiple interconnected lstm layers.
    One may interact with this class or with the lstm layer class directly.
    """
    class network(object):

        """
        Create the lstm neural network.
        Pass layer parameters into the layer constructor function.
        """
        def __init__(self, params):
            self.layers = [lstm.layer(lparams)for lparams in params['layers']]

        """
        Pass the stimulus forward through each layer in the network.
        """
        def forward(self, stimulus):
            self.activation = stimulus
            for i in range(len(self.layers)):
                self.activation = self.layers[i].forward(self.activation)
            return self.activation
        
        """
        Prepare the lstm layers for backward propagation through time.
        """
        def initBackward(self):
            for i in range(len(self.layers)):
                self.layers[i].initBackward()

        """
        Pass the error backward through each layer in the network.
        """
        def backward(self, error):
            self.delta = error
            for i in reversed(range(len(self.layers))):
                self.delta = self.layers[i].backward(self.delta)
            return self.delta

    """
    This class represents a neural network layer, which accepts an input, produces an output, and can backpropogate error in order to learn.
    The lstm algorithm uses backpropagation through time across the entire input sequence.
    """
    class layer:

        """
        Reset the algorithm after a full forward and backward pass cycle has been completed, and the unrolled network has been collapsed.
        """
        def reset(self):
            self.unrolled = []
            self.time = -1
            self._cell.reset()

        """
        Initialize the hidden layer, and create the lstm memory cell.
        """
        def __init__(self, params):
            self.alpha = params['alpha']
            self._cell = lstm.cell(params)
            self.reset()

        """
        Compute a single forward pass and obtain an output.
        """
        def forward(self, stimulus):
            self.time += 1
            self.output = self._cell.forward(stimulus)
            self.unrolled.append(self._cell)
            return self.output

        """
        Prepare the algorithm to begin backpropogation through time, and update the weight parameters.
        All changes to trainable parameters are summed onto the initial value of the parameters.
        """
        def initBackward(self):
            self.unrolled[0].cellBias *= -self.time
            self.unrolled[0].cellStimulusWeights *= -self.time
            self.unrolled[0].cellStateWeights *= -self.time
            self.unrolled[0].cellRecurrentWeights *= -self.time

            self.unrolled[0].inputBias *= -self.time
            self.unrolled[0].inputStimulusWeights *= -self.time
            self.unrolled[0].inputStateWeights *= -self.time
            self.unrolled[0].inputRecurrentWeights *= -self.time

            self.unrolled[0].forgetBias *= -self.time
            self.unrolled[0].forgetStimulusWeights *= -self.time
            self.unrolled[0].forgetStateWeights *= -self.time
            self.unrolled[0].forgetRecurrentWeights *= -self.time

            self.unrolled[0].outputBias *= -self.time
            self.unrolled[0].outputStimulusWeights *= -self.time
            self.unrolled[0].outputStateWeights *= -self.time
            self.unrolled[0].outputRecurrentWeights *= -self.time

        """
        Compute a single backward pass and obtain a delta to send to earlier hidden layers in the network.
        """
        def backward(self, error):
            self.delta = self.unrolled[self.time].backward(error, self.alpha)
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

            """
            Pass error backward through time to another instance of the memory cell.
            """
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
    All neural computation is calculated as linear algebra matrix multipliation.
    """
    class cell:

        """
        This section creates the matrices used by this algorithm, includig the weight matrices, and an adjacency matrix.
        """
        def create(self):
            self.reset()

            """
            The bias term for each gated sum in the memory cell.
            The size of the bias should be equivalent to the size of the memory cell.
            """
            self.cellBias = np.random.normal(0, 1, (1, self.outputWidth))
            self.inputBias = np.random.normal(0, 1, (1, self.outputWidth))
            self.forgetBias = np.random.normal(0, 1, (1, self.outputWidth))
            self.outputBias = np.random.normal(0, 1, (1, self.outputWidth))

            """
            These weights are used to compute a weighted multiplication with the input to the memory cell.
            Matrix multiplication transforms the input of size (1, self.inputWidth) to an output of size (1, self.outputSize).
            """
            self.cellStimulusWeights = np.random.normal(0, 1, (self.inputWidth, self.outputWidth))
            self.inputStimulusWeights = np.random.normal(0, 1, (self.inputWidth, self.outputWidth))
            self.forgetStimulusWeights = np.random.normal(0, 1, (self.inputWidth, self.outputWidth))
            self.outputStimulusWeights = np.random.normal(0, 1, (self.inputWidth, self.outputWidth))
            
            """
            These weights are used to compute a weighted multiplication with the memory state of the memory cell.
            Matrix multiplication transforms the memory state of size (1, self.outputWidth) to an output of size (1, self.outputSize).
            """
            self.cellStateWeights = np.random.normal(0, 1, (self.outputWidth, self.outputWidth))
            self.inputStateWeights = np.random.normal(0, 1, (self.outputWidth, self.outputWidth))
            self.forgetStateWeights = np.random.normal(0, 1, (self.outputWidth, self.outputWidth))
            self.outputStateWeights = np.random.normal(0, 1, (self.outputWidth, self.outputWidth))

            """
            These weights are used to compute a weighted multiplication with the cell output of the memory cell.
            Matrix multiplication transforms the cell output of size (1, self.outputWidth) to an output of size (1, self.outputSize).
            """
            self.cellRecurrentWeights = np.random.normal(0, 1, (self.outputWidth, self.outputWidth))
            self.inputRecurrentWeights = np.random.normal(0, 1, (self.outputWidth, self.outputWidth))
            self.forgetRecurrentWeights = np.random.normal(0, 1, (self.outputWidth, self.outputWidth))
            self.outputRecurrentWeights = np.random.normal(0, 1, (self.outputWidth, self.outputWidth))

        """
        This section will reset the lstm cell after a complete backpropapgation cycle has occured.
        """
        def reset(self):

            """
            These variables are used during forward propagation.
            Each variable represents a nopde within an lstm memory cell.
            """
            self.stimulus = np.zeros((1, self.inputWidth))
            self.inputGate = np.zeros((1, self.outputWidth))
            self.forgetGate = np.zeros((1, self.outputWidth))
            self.outputGate = np.zeros((1, self.outputWidth))
            self.cellInput = np.zeros((1, self.outputWidth))
            self.cellOutput = np.zeros((1, self.outputWidth))
            self.previousState = np.zeros((1, self.outputWidth))
            self.currentState = np.zeros((1, self.outputWidth))
            
            """
            These variables are stored and calculated during forward propogation at each update.
            Real-time rercurrent learning describes how error gradients are calculated and propagated forward at eahc update cycle.
            Note how the partial derivative for the output gate need not be stored as such does not pass through the memory state.
            """

            """
            These partial derivatives represent error traveling through the memory state, and through the cell input.
            """
            self.cellStimulusDelta = np.zeros((1, self.inputWidth))
            self.cellStateDelta = np.zeros((1, self.outputWidth))
            self.cellRecurrentDelta = np.zeros((1, self.outputWidth))
            self.cellBiasPartial = np.zeros((1, self.outputWidth))
            self.cellStimulusPartial = np.zeros((self.inputWidth, self.outputWidth))
            self.cellStatePartial = np.zeros((self.outputWidth, self.outputWidth))
            self.cellRecurrentPartial = np.zeros((self.outputWidth, self.outputWidth))
            
            """
            These partial derivatives represent error traveling through the memory state, and through the input gate.
            """
            self.inputStimulusDelta = np.zeros((1, self.inputWidth))
            self.inputStateDelta = np.zeros((1, self.outputWidth))
            self.inputRecurrentDelta = np.zeros((1, self.outputWidth))
            self.inputBiasPartial = np.zeros((1, self.outputWidth))
            self.inputStimulusPartial = np.zeros((self.inputWidth, self.outputWidth))
            self.inputStatePartial = np.zeros((self.outputWidth, self.outputWidth))
            self.inputRecurrentPartial = np.zeros((self.outputWidth, self.outputWidth))
            
            """
            These partial derivatives represent error traveling through the memory state, and through the forget gate.
            """
            self.forgetStimulusDelta = np.zeros((1, self.inputWidth))
            self.forgetStateDelta = np.zeros((1, self.outputWidth))
            self.forgetRecurrentDelta = np.zeros((1, self.outputWidth))
            self.forgetBiasPartial = np.zeros((1, self.outputWidth))
            self.forgetStimulusPartial = np.zeros((self.inputWidth, self.outputWidth))
            self.forgetStatePartial = np.zeros((self.outputWidth, self.outputWidth))
            self.forgetRecurrentPartial = np.zeros((self.outputWidth, self.outputWidth))

            """
            Reset the accumulating error partial derivatives.
            """
            self.recurrentError = np.zeros((1, self.outputWidth))
            self.stateError = np.zeros((1, self.outputWidth))

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

            """
            This section will compute a single forward pass for this lstm memory cell.
            """
            self.stimulus = stimulus
            self.inputGate = lstm.gate(
                np.dot(self.stimulus, self.inputStimulusWeights) +
                np.dot(self.currentState, self.inputStateWeights) +
                np.dot(self.cellOutput, self.inputRecurrentWeights) +
                self.inputBias)
            self.forgetGate = lstm.gate(
                np.dot(self.stimulus, self.forgetStimulusWeights) +
                np.dot(self.currentState, self.forgetStateWeights) + 
                np.dot(self.cellOutput, self.forgetRecurrentWeights) +
                self.forgetBias)
            self.outputGate = lstm.gate(
                np.dot(self.stimulus, self.outputStimulusWeights) +
                np.dot(self.currentState, self.outputStateWeights) +
                np.dot(self.cellOutput, self.outputRecurrentWeights) +
                self.outputBias)
            self.cellInput = lstm.activate(
                np.dot(self.stimulus, self.cellStimulusWeights) +
                np.dot(self.currentState, self.cellStateWeights) +
                np.dot(self.cellOutput, self.cellRecurrentWeights) +
                self.cellBias)
            self.previousState = self.currentState
            self.currentState = (self.forgetGate * self.previousState) + (self.inputGate * self.cellInput)
            self.cellOutput = lstm.activate(self.currentState)

            """
            The following sections will compute the error partial derivatives through the memory state.
            the forward propogation of error partial derivatives constituted real-time recurrent learning.
            """

            """
            Compute the partial derivative of the memory state with respect to the cell stimulus, the memory state, and the recurrent output through the cell input.
            Also, compute the partial derivative of the memory state with respect to the cell stimulus weights, the memory state weights, and the recurrent output weights through the cell input.
            """
            self.cellStimulusDelta = np.sum(np.dot(self.cellStimulusDelta.transpose(), self.forgetGate), axis=1).reshape((1, self.inputWidth)) + \
                np.dot((self.inputGate * lstm.aprime(self.cellInput)), self.cellStimulusWeights.transpose())
            self.cellStateDelta = np.sum(np.dot(self.cellStateDelta.transpose(), self.forgetGate), axis=1).reshape((1, self.outputWidth)) + \
                np.dot((self.inputGate * lstm.aprime(self.cellInput)), self.cellStateWeights.transpose())
            self.cellRecurrentDelta = np.sum(np.dot(self.cellRecurrentDelta.transpose(), self.forgetGate), axis=1).reshape((1, self.outputWidth)) + \
                np.dot((self.inputGate * lstm.aprime(self.cellInput)), self.cellRecurrentWeights.transpose())
            self.cellBiasPartial = (self.cellBiasPartial * self.forgetGate) + (self.inputGate * lstm.aprime(self.cellInput))
            self.cellStimulusPartial = (self.cellStimulusPartial * self.forgetGate) + \
                np.dot(self.stimulus.transpose(), (self.inputGate * lstm.aprime(self.cellInput)))
            self.cellStatePartial = (self.cellStatePartial * self.forgetGate) + \
                np.dot(self.previousState.transpose(), (self.inputGate * lstm.aprime(self.cellInput)))
            self.cellRecurrentPartial = (self.cellRecurrentPartial * self.forgetGate) + \
                np.dot(self.cellOutput.transpose(), (self.inputGate * lstm.aprime(self.cellInput)))

            """
            Compute the partial derivative of the memory state with respect to the cell stimulus, the memory state, and the recurrent output through the input gate.
            Also, compute the partial derivative of the memory state with respect to the cell stimulus weights, the memory state weights, and the recurrent output weights through the input gate.
            """
            self.inputStimulusDelta = np.sum(np.dot(self.inputStimulusDelta.transpose(), self.forgetGate), axis=1).reshape((1, self.inputWidth)) + \
                np.dot((self.cellInput * lstm.gprime(self.inputGate)), self.inputStimulusWeights.transpose())
            self.inputStateDelta = np.sum(np.dot(self.inputStateDelta.transpose(), self.forgetGate), axis=1).reshape((1, self.outputWidth)) + \
                np.dot((self.cellInput * lstm.gprime(self.inputGate)), self.inputStateWeights.transpose())
            self.inputRecurrentDelta = np.sum(np.dot(self.inputRecurrentDelta.transpose(), self.forgetGate), axis=1).reshape((1, self.outputWidth)) + \
                np.dot((self.cellInput * lstm.gprime(self.inputGate)), self.inputRecurrentWeights.transpose())
            self.inputBiasPartial = (self.inputBiasPartial * self.forgetGate) + (self.cellInput * lstm.gprime(self.inputGate))
            self.inputStimulusPartial = (self.inputStimulusPartial * self.forgetGate) + \
                np.dot(self.stimulus.transpose(), (self.cellInput * lstm.gprime(self.inputGate)))
            self.inputStatePartial = (self.inputStatePartial * self.forgetGate) + \
                np.dot(self.previousState.transpose(), (self.cellInput * lstm.gprime(self.inputGate)))
            self.inputRecurrentPartial = (self.inputRecurrentPartial * self.forgetGate) + \
                np.dot(self.cellOutput.transpose(), (self.cellInput * lstm.gprime(self.inputGate)))

            """
            Compute the partial derivative of the memory state with respect to the cell stimulus, the memory state, and the recurrent output through the forget gate.
            Also, compute the partial derivative of the memory state with respect to the cell stimulus weights, the memory state weights, and the recurrent output weights through the forget gate.
            """
            self.forgetStimulusDelta = np.sum(np.dot(self.forgetStimulusDelta.transpose(), self.forgetGate), axis=1).reshape((1, self.inputWidth)) + \
                np.dot((self.cellOutput * lstm.gprime(self.forgetGate)), self.forgetStimulusWeights.transpose())
            self.forgetStateDelta = np.sum(np.dot(self.forgetStateDelta.transpose(), self.forgetGate), axis=1).reshape((1, self.outputWidth)) + \
                np.dot((self.cellOutput * lstm.gprime(self.forgetGate)), self.forgetStateWeights.transpose())
            self.forgetRecurrentDelta = np.sum(np.dot(self.forgetRecurrentDelta.transpose(), self.forgetGate), axis=1).reshape((1, self.outputWidth)) + \
                np.dot((self.cellOutput * lstm.gprime(self.forgetGate)), self.forgetRecurrentWeights.transpose())
            self.forgetBiasPartial = (self.forgetBiasPartial * self.forgetGate) + (self.cellOutput * lstm.gprime(self.forgetGate))
            self.forgetStimulusPartial = (self.forgetStimulusPartial * self.forgetGate) + \
                np.dot(self.stimulus.transpose(), (self.cellOutput * lstm.gprime(self.forgetGate)))
            self.forgetStatePartial = (self.forgetStatePartial * self.forgetGate) + \
                np.dot(self.previousState.transpose(), (self.cellOutput * lstm.gprime(self.forgetGate)))
            self.forgetRecurrentPartial = (self.forgetRecurrentPartial * self.forgetGate) + \
                np.dot(self.cellOutput.transpose(), (self.cellOutput * lstm.gprime(self.forgetGate)))
            return self.outputGate * self.cellOutput

        """
        Compute a single backward pass within this serriform cell, and output a weighted error for earlier hidden layers.
        """
        def backward(self, error, alpha):
            delta = (error + self.recurrentError)

            """
            Update the output gate weights, and pass error partial derivatives backward through the output gate connections.
            """
            recurrentDelta = np.dot((delta * self.cellOutput * lstm.gprime(self.outputGate)), self.outputRecurrentWeights.transpose())
            stateDelta = np.dot((delta * self.cellOutput * lstm.gprime(self.outputGate)), self.outputStateWeights.transpose())
            stimulusDelta = np.dot((delta * self.cellOutput * lstm.gprime(self.outputGate)), self.outputStimulusWeights.transpose())
            self.outputBias -= alpha * delta * self.cellOutput * lstm.gprime(self.outputGate)
            self.outputStimulusWeights -= alpha * np.dot(self.stimulus.transpose(), (delta * self.cellOutput * lstm.gprime(self.outputGate)))
            self.outputStateWeights -= alpha * np.dot(self.previousState.transpose(), (delta * self.cellOutput * lstm.gprime(self.outputGate)))
            self.outputRecurrentWeights -= alpha * np.dot(self.cellOutput.transpose(), (delta * self.cellOutput * lstm.gprime(self.outputGate)))

            """
            Pass the delta backward through the cell output using the chain rule.
            """
            delta *= self.outputGate * lstm.aprime(self.cellOutput)
            delta += self.stateError

            """
            Update the forget gate weights, and pass error partial derivatives backward through the forget gate gate connections.
            Note that the partial derivatives calculated using real-time recurrent learning are used in this calculation.
            """
            recurrentDelta += np.sum(np.dot(self.forgetRecurrentDelta.transpose(), delta), axis=1).reshape((1, self.outputWidth))
            stateDelta += np.sum(np.dot(self.forgetStateDelta.transpose(), delta), axis=1).reshape((1, self.outputWidth))
            stimulusDelta += np.sum(np.dot(self.forgetStimulusDelta.transpose(), delta), axis=1).reshape((1, self.inputWidth))
            self.forgetBias -= alpha * delta * self.forgetBiasPartial
            self.forgetRecurrentWeights -= alpha * delta * self.forgetRecurrentPartial
            self.forgetStateWeights -= alpha * delta * self.forgetStatePartial
            self.forgetStimulusWeights -= alpha * delta * self.forgetStimulusPartial
            
            """
            Update the input gate weights, and pass error partial derivatives backward through the input gate connections.
            Note that the partial derivatives calculated using real-time recurrent learning are also used in this calculation.
            """
            recurrentDelta += np.sum(np.dot(self.inputRecurrentDelta.transpose(), delta), axis=1).reshape((1, self.outputWidth))
            stateDelta += np.sum(np.dot(self.inputStateDelta.transpose(), delta), axis=1).reshape((1, self.outputWidth))
            stimulusDelta += np.sum(np.dot(self.inputStimulusDelta.transpose(), delta), axis=1).reshape((1, self.inputWidth))
            self.inputBias -= alpha * delta * self.inputBiasPartial
            self.inputRecurrentWeights -= alpha * delta * self.inputRecurrentPartial
            self.inputStateWeights -= alpha * delta * self.inputStatePartial
            self.inputStimulusWeights -= alpha * delta * self.inputStimulusPartial

            """
            Update the cell input weights, and pass error partial derivatives backward through the cell input connections.
            Note that the partial derivatives calculated using real-time recurrent learning are also used in this calculation.
            """
            recurrentDelta += np.sum(np.dot(self.cellRecurrentDelta.transpose(), delta), axis=1).reshape((1, self.outputWidth))
            stateDelta += np.sum(np.dot(self.cellStateDelta.transpose(), delta), axis=1).reshape((1, self.outputWidth))
            stimulusDelta += np.sum(np.dot(self.cellStimulusDelta.transpose(), delta), axis=1).reshape((1, self.inputWidth))
            self.cellBias -= alpha * delta * self.cellBiasPartial
            self.cellRecurrentWeights -= alpha * delta * self.cellRecurrentPartial
            self.cellStateWeights -= alpha * delta * self.cellStatePartial
            self.cellStimulusWeights -= alpha * delta * self.cellStimulusPartial

            """
            Update the saved error partial derivates with the combined sum of error partial derivates from the output gate, forget gate, input gate, and cell input.
            """
            self.recurrentDelta = recurrentDelta
            self.stateDelta = stateDelta
            return stimulusDelta

"""
This function serves as the entry point for this python applicaion.
The lstm layer created above is created, and tested on fake data.
"""
def main():

    net = lstm.network({
        'layers': [
            {'inputWidth': 1, 'outputWidth': 1, 'alpha': 0.01}
        ]
    })

    for y in range(128):
        output = net.forward(np.array([[0.8]]))
        net.initBackward()
        net.backward((output - np.array([[0.8]])))
        print(output)

main()


###
### Problem with outer product in the partial derivative calculation
###