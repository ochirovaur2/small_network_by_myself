import numpy as np

def segmoid(x):
	return 1 / (1 + np.exp(-x))

def get_derivative_of_the_segmoid(x):
	return x * ( 1 - x)


input_data = np.array([[0, 0, 1], [1, 1, 1], [1, 0, 1], [0, 1, 1]])

output_data = np.array( [[0, 1, 1, 0]] ).T


weights_1_1 = 2 * np.random.random_sample((3, 1)) - 1
weights_1_2 = 2 * np.random.random_sample((3, 1)) - 1

learing_rate = 0.01



for i in range(10000):
	num_i = 0
	weights_1_1_delta = []

	for iter in input_data:

		res = segmoid(np.dot(input_data[num_i], weights_1_1)) 
		error_of_layer_1_1 = res - output_data[num_i]

		weights_1_1_delta.append(get_derivative_of_the_segmoid(res) * error_of_layer_1_1)
		num_i+=1

	weights_1_1_delta = np.array(weights_1_1_delta) 
	weights_1_1 -= np.dot(input_data.T, weights_1_1_delta)
	

print("Weights after training")
print(weights_1_1)
print(segmoid(np.dot([1,1 ,0], weights_1_1)))


# layer_1_1 = segmoid(np.dot(input_data, weights_1_1))
# layer_1_2 = segmoid(np.dot(input_data, weights_1_2))


# error_of_layer_1_1 = layer_1_1 - output_data
# error_of_layer_1_2 = layer_1_2 - output_data

# input_of_layer_2_1 = np.array([[layer_1_1, layer_1_2]])
# # print(input_of_layer_2_1)
# # layer_2_1 = segmoid(np.dot(input_data, weights_1_1))
# # weights_1_1_delta = get_derivative_of_the_segmoid(res) * error

# # weights_1_1 -= np.dot(input_data.T, weights_1_1_delta) * learing_rate



# # 

