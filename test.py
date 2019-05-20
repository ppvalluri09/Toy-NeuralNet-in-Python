from nn import NeuralNetwork as nn

brain = nn(2, 2, 1)
data = [1, 1]
target = [1]
prediction = brain.predict(data)
print(prediction)

for x in range(0, 1000):
	brain.feedForward(data, target)

prediction = brain.predict(data)
print(prediction)