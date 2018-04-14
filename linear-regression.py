import random
import matplotlib.pyplot as plt
from utils import draw_line, draw_plot

def generate_points(n):
    interval = 2
    points = ([], [])
    max = interval
    for x in range(n):
        x = random.randint(max - interval, max)
        y = random.randint(max - interval, max)
        points[0].append(x)
        points[1].append(y)
        max += interval

    return points

L_RATE = 0.0000001
theta0 = 10
theta1 = 10


def h(x):
    return theta0 + theta1 * x

def cost_func(x, y):
    size = len(x)
    # cost is sum((h(Xi) - Yi))^2 / 2m
    cost = 0
    for i in range(size):
        cost += (h(x[i]) - y[i]) ** 2
    cost /= (2 * size)
    return cost

def update_parameters(x, y):
    global theta0, theta1
    deviation0 = 0
    deviation1 = 0
    size = len(x)
    for i in range(size):
        deviation0 += (h(x[i]) - y[i])
        deviation1 += ((h(x[i]) - y[i]) * x[i])

    deviation0 = (deviation0 * L_RATE) / size
    deviation1 = (deviation1 * L_RATE) / size

    theta0 -= deviation0
    theta1 -= deviation1


iterations = 10000
cost_list = [[], []]
x, y = generate_points(50)
for i in range(iterations):
    update_parameters(x, y)
    if(i % 100 == 0):
        cost_list[0].append(i)
        cost_list[1].append(cost_func(x, y))

# Draw plot of cost vs number of iterations
# iterations, cost = cost_list
# draw_plot(plt, iterations, 'Iterations', cost, 'Cost', 'line')
# plt.show()

# To show the fitted line and data points
draw_plot(plt, x, 'Size of house', y, 'Cost of house', 'dot')
xs = [0, max(x)]
ys = [h(xs[0]), h(xs[1])]
draw_line(plt, xs, ys)
plt.show()
