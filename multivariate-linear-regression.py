
import numpy as np
import csv
import matplotlib.pyplot as plt

n = 16 # number of input features.
m = 60 # number of training examples.
grad = np.zeros(shape = (n, 1))
theta = np.ones(shape=(n, 1), dtype = float)
hx = np.ones(shape=(m, 1), dtype = float)

file_handle = open("datasets/air-pollution/data.csv", "r")
reader = csv.reader(file_handle, delimiter = ',')

learning_rate = 1e-6

def h(X):
	global theta
	res = np.matmul(np.transpose(theta), X)
	return res

cost_list = []
itr_list = []

def gradient_descent_algorithm():
	global theta, grad
	num_itrs = 10000
	for itr in range(num_itrs):
		file_handle.seek(0)
		total_cost = 0.0
		idx = 0
		for row in reader:
			X = [float(x) for x in row[0: -1]]
			# list of floats
			X = np.asarray(X)
			np.reshape(X, [n, 1])
			hx[idx][0] = h(X)
			y_correct = float(row[0])
			diff = (hx[idx][0] - y_correct)
			total_cost += (diff * diff)
			idx += 1
		for j in range(n):
			grad[j][0] = 0.0
			i = 0
			file_handle.seek(0)
			for row in reader:
				y_correct = float(row[-1])
				xij = float(row[j + 1])
				diff = hx[i][0] - y_correct
				grad[j][0] += ((learning_rate * diff * xij) / m)
				i += 1

		theta = theta - grad
		total_cost = total_cost /(2 * m)
		cost_list.append(total_cost)
		itr_list.append(itr + 1)

gradient_descent_algorithm()


plt.plot(itr_list, cost_list, label = "cost")

plt.xlabel("iterations")
# naming the y axis
plt.ylabel('Cost')
# giving a title to my graph
plt.title('Cost vs iterations')
 
# show a legend on the plot
plt.legend()
 
# function to show the plot
plt.show()




ypaxis = []
ycaxis = []
xaxis = []

index = 0
file_handle.seek(0)
for row in reader:
	X = [float(x) for x in row[1:]]
	# list of floats
	X = np.asarray(X)
	np.reshape(X, [n, 1])
	pred = h(X)
	y_correct = float(row[0])
	index += 1
	ypaxis.append(pred)
	ycaxis.append(y_correct)
	xaxis.append(index)


plt.plot(xaxis, ycaxis, label = "correct")
plt.plot(xaxis, ypaxis, label = "prediction")
plt.xlabel("examples")
# naming the y axis
plt.ylabel('h_theta')
plt.title('correct vs predicted')

# show a legend on the plot
plt.legend()
 
# function to show the plot
plt.show()
