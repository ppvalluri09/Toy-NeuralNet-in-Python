import numpy
import math

def sigmoid(x, derivative=False):
    sigm = 1. / (1. + numpy.exp(-x))
    if derivative:
        return sigm * (1. - sigm)
    return sigm

def squeeze(data, derivative=False):
	if derivative:
		for rows in range(0, len(data)):
			for cols in range(0, len(data[0])):
				data[rows][cols] = sigmoid(data[rows][cols], True)
	else:
		for rows in range(0, len(data)):
			for cols in range(0, len(data[0])):
				data[rows][cols] = sigmoid(data[rows][cols])

	return data

def fromArray(_list):
	values = numpy.random.rand(len(_list), 1)
	for rows in range(0, len(_list)):
		values[rows][0] = _list[rows]

	return values


class NeuralNetwork:

	# constructor for making the Neural Network
	def __init__(self, input_nodes, hidden_nodes, output_nodes):

		#setting the input, hidden and output nodes
		self.input_nodes = input_nodes
		self.hidden_nodes = hidden_nodes
		self.output_nodes = output_nodes

		# randomizing thee= weights
		self.weights_ih = numpy.random.uniform(-1, 1, (self.hidden_nodes, self.input_nodes))
		self.weights_ho = numpy.random.uniform(-1, 1, (self.output_nodes, self.hidden_nodes))
		# self.weights_ih = numpy.random.rand(self.hidden_nodes, self.input_nodes)
		# self.weights_ho = numpy.random.rand(self.output_nodes, self.hidden_nodes)

		# randomizing the bias
		self.bias_h = numpy.random.uniform(-1, 1, (self.hidden_nodes, 1))
		self.bias_o = numpy.random.uniform(-1, 1, (self.output_nodes, 1))
		# self.bias_h = numpy.random.rand(self.hidden_nodes, 1)
		# self.bias_o = numpy.random.rand(self.output_nodes, 1)

		self.learning_rate = 0.1

	def show(self):
		print(self.weights_ih)
		print('')
		print(self.weights_ho)
		print('')
		print(self.bias_h)
		print('')
		print(self.bias_o)
		print('')

	def predict(self, data):
		inputs = numpy.array(fromArray(data))
		hidden = numpy.matmul(self.weights_ih, inputs)
		hidden = numpy.add(hidden, self.bias_h)

		# Activation Function
		hidden = squeeze(hidden)

		output = numpy.dot(self.weights_ho, hidden)
		output = numpy.add(output, self.bias_o)
		# Activation Function
		output = squeeze(output)

		return numpy.array(output)

	def feedForward(self, data, targets):
		inputs = numpy.array(fromArray(data))
		hidden = numpy.matmul(self.weights_ih, inputs)
		hidden = numpy.add(hidden, self.bias_h)

		# Activation Function
		hidden = squeeze(hidden)

		output = numpy.dot(self.weights_ho, hidden)
		output = numpy.add(output, self.bias_o)
		# Activation Function
		output = squeeze(output)

		# Training
		targets = numpy.array(fromArray(targets))
		output_errors = numpy.subtract(targets, output)

		gradients = squeeze(output, True)
		gradients = numpy.multiply(gradients, output_errors)
		gradients = numpy.multiply(gradients, self.learning_rate)

		# Calculating Deltas
		hidden_t = hidden.T
		weight_ho_deltas = numpy.dot(gradients, hidden_t)

		# Adjust the bias by its deltas
		self.weight_ho = numpy.add(self.weights_ho, weight_ho_deltas)
		self.bias_o = numpy.add(self.bias_o, gradients)


		# Calculate hidden layer errors
		who_t = self.weights_ho.T
		hidden_errors = numpy.dot(who_t, output_errors)

		# Calculate hidden layer gradient
		hidden_gradient = squeeze(hidden, True)
		hidden_gradient = numpy.multiply(hidden_gradient, hidden_errors)
		hidden_gradient = numpy.multiply(hidden_gradient, self.learning_rate)

		# Calculate input->hidden deltas
		inputs_t = inputs.T
		weight_ih_deltas = numpy.dot(hidden_gradient, inputs_t)

		# Adjust the weight and bias
		self.weights_ih = numpy.add(self.weights_ih, weight_ih_deltas)
		self.bias_h = numpy.add(self.bias_h, hidden_gradient)


