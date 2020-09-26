import numpy as np

def segmoid(x):
	return 1 / (1 + np.exp(-x))

def get_derivative_of_the_segmoid(x):
	return x * ( 1 - x)


input_data = np.array([[0, 0, 1], [1, 1, 1], [1, 0, 1], [0, 1, 1]])

output_data = np.array( [[0, 1, 1, 0]] ).T


weights_1_1 = 2 * np.random.random_sample((3, 1)) - 1

print(weights_1_1)
learing_rate = 0.01

for iter in range(10000):
	res = segmoid(np.dot(input_data, weights_1_1))

	error = res - output_data

	weights_1_1_delta = get_derivative_of_the_segmoid(res) * error

	weights_1_1 -= np.dot(input_data.T, weights_1_1_delta) * learing_rate



print(segmoid(np.dot([1,1 ,0], weights_1_1)))

