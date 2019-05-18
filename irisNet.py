#ems236 P2
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as la
import re
import math

################################################################
#1a)

classNameDict = {
	"setosa" : 0
	, "versicolor": 1
	, "virginica": 2
}

# didn't want to have to change in multiple places when I spell the names wrong
# also did not actually end up using these
SETOSA = "setosa"
VERISICOLOR = "versicolor"
VIRGINICA = "virginica"

#define attribute names so I don't have to remember the indeces
#don't use very often but don't want to delete
SLENGTH = 0
SWIDTH = 1
PLENGTH = 2
PWIDTH = 3

#read all data.
file = open("irisdata.csv", "r")
doneOne = False
#initialize the array with a row of zeros because I'm bad at python
data = np.zeros((1, 5))
class1 = np.zeros((1, 4))
class2 = np.zeros((1, 4))
class3 = np.zeros((1, 4))

for line in file:
	#take all whitespace out because it's annoying
	vals = re.sub(r'\s+', '', line)
	vals = vals.split(",")
	if doneOne is True:
		data = np.vstack((data, [float(vals[0]), float(vals[1]), float(vals[2]), float(vals[3]), int(classNameDict[vals[4]])]))
		if vals[4] == SETOSA:
			class1 = np.vstack((class1, [float(vals[0]), float(vals[1]), float(vals[2]), float(vals[3])]))
		elif vals[4] == VERISICOLOR:
			class2 = np.vstack((class2, [float(vals[0]), float(vals[1]), float(vals[2]), float(vals[3])]))
		else:
			class3 = np.vstack((class3, [float(vals[0]), float(vals[1]), float(vals[2]), float(vals[3])]))
	else:
		#ignore first row labeling columns.
		doneOne = True
file.close()

#get those zeros out of there
data = np.delete(data, 0, 0)
class1 = np.delete(class1, 0, 0)
class2 = np.delete(class2, 0, 0)
class3 = np.delete(class3, 0, 0)

#graph petal length vs width for class 2 and 3
plt.figure(1)
plt.scatter(class2[:,PLENGTH], class2[:,PWIDTH], color='blue', marker="2")
plt.scatter(class3[:,PLENGTH], class3[:,PWIDTH], color='green', marker="+")
plt.xlabel("Petal Length")
plt.ylabel("Petal Width")
plt.title("Petal Length vs Width for class 2 and 3")
plt.tight_layout()


################################################################
#1b)
#Make linear classifier with logistic
#a logistic function that works with arrays
def logistic(x):
	return (1/(1+np.exp(-1 * x)))

#expects weight to include some w0, the bias.
def WeightByData(weight, data):
	return weight[0] + np.inner(weight[1:], data)

#classify a single data vector x
def classify(weight, x):
	if logistic(WeightByData(weight, x)) >= 0.5:
		return 1
	else:
		return 0

#generate the decision boundary from weights on the petal domain
def decisionBoundary(weight):
	X = np.linspace(2.5, 7, 100)
	Y = (-1*weight[1]/weight[2] * X) - (weight[0]/weight[2])
	return X, Y

################################################################
#1c)
weight = np.array([-6.45, 1., 1.])


#plot decision boundary
plt.figure(2)
plt.scatter(class2[:,PLENGTH], class2[:,PWIDTH], color='blue', marker="2")
plt.scatter(class3[:,PLENGTH], class3[:,PWIDTH], color='green', marker="+")
plt.xlabel("Petal Length")
plt.ylabel("Petal Width")

lineX, lineY = decisionBoundary(weight)
#lineX = np.linspace(2.5, 7, 100)
#lineY = (-1*weight[1]/weight[2] * lineX) - weight[0]
plt.plot(lineX, lineY, color='red')
plt.ylim((0.8, 2.7))
plt.xlim((2.5, 7)) 
plt.title("A Hand-Selected Decison Boundary")

################################################################
#1d)
#plot sigmoid in 3d
fig = plt.figure(3)
ax = fig.add_subplot(111, projection="3d")
x = np.linspace(2.5, 7, 1000)
y = np.linspace(0.8, 2.7, 1000)

#Do some weird magic with dimensionality of data see https://stackoverflow.com/questions/9170838/surface-plots-in-matplotlib
bigX, bigY = np.meshgrid(x, y)
#More weird dimensionality magic
zs = np.array([logistic(weight[0] + x * weight[1] + y * weight[2]) for x,y in zip(np.ravel(bigX), np.ravel(bigY))])
z = zs.reshape(bigX.shape)
ax.plot_surface(bigX, bigY, z, color='b')
ax.set_xlabel("x1 (petal length)")
ax.set_ylabel("x2 (petal width)")
ax.set_zlabel("logistic(a(x))")
plt.title("The Output of the Neural Network")


#################################################################
#1e)
#classify some example datapoints
testData = np.array([
[3.5, 1]
,[4.9, 1.5]
,[4.5, 1.7]
,[6.7, 2.]
])
output = logistic(WeightByData(weight, testData))
plt.figure(4)
plt.scatter(testData[:2,0], testData[:2,1], color='blue', marker="2")
plt.scatter(testData[2:,0], testData[2:,1], color='green', marker="+")
plt.xlabel("Petal Length")
plt.ylabel("Petal Width")

lineX, lineY = decisionBoundary(weight)
plt.plot(lineX, lineY, color='red')
plt.ylim((0.8, 2.7))
plt.xlim((2.5, 7)) 
plt.title("Selected Example Classifications")


#for x in range(len(petalData)):
#	#print(classify(weight, petalData[x]))
#	y = x+1

################################################################
#2a)
#data is an array of 0 or 1, definining which class it belongs to (0 = class2, 1 = class3)
#weight is a weight vector
#patterns is a set of data pairs
def totSquareError(data, weight, patterns):
	# E = 1/2(sum(y(xi, w) - pattern)^2)
	output = logistic(WeightByData(weight, patterns))
	#the error for each data point
	stepError = output - data
	return 0.5 * np.sum(stepError ** 2)

def meanSquareError(data, weight, patterns):
	# E = 1/2n(sum(y(xi, w) - pattern)^2)
	output = logistic(WeightByData(weight, patterns))
	#the error for each data point
	stepError = output - data
	return 0.5 * np.sum(stepError ** 2) / len(data)

################################################################
#2b)
#get Pattern Info
#Does not seem like the easiest way to do this.
petalData = np.hstack((np.zeros((1, len(class2))), np.ones((1, len(class3)))))[0]

#throw out columns that don't matter.
patterns = np.vstack((class2, class3))[:, 2:]

#set weights
weight1 = np.array([-6.45, 1., 1.])
weight2 = np.array([2, -0.8, 1.])

W1X, W1Y = decisionBoundary(weight1)
W2X, W2Y = decisionBoundary(weight2)

plt.figure(5)
plt.scatter(class2[:,PLENGTH], class2[:,PWIDTH], color='blue', marker="2")
plt.scatter(class3[:,PLENGTH], class3[:,PWIDTH], color='green', marker="+")
plt.xlabel("Petal Length")
plt.ylabel("Petal Width")

plt.plot(W1X, W1Y, label="A Low-Error Boundary", color='red')
plt.plot(W2X, W2Y, label="A High-Error Boundary", color='black')
plt.legend()
#plt.ylim((0.8, 2.7))
#plt.xlim((2.5, 7)) 
plt.title("Multiple Decision Boundaries")
print("2d: weight1 error: " + str(meanSquareError(petalData, weight1, patterns)))
print("2d: weight2 error: " + str(meanSquareError(petalData, weight2, patterns)))

################################################################
#2e)
#data is an array of 0 or 1, definining which class it belongs to (0 = class2, 1 = class3)
#weight is a weight vector
#patterns is a set of data pairs
def gradientError(data, weight, patterns):
	#logistic(wT x + w0)
	sigmoidData = logistic(WeightByData(weight, patterns))
	#logistic(wT x + w0) - c
	error = sigmoidData - data
	#calculate all terms before the vector multiplication
	coefficient = error * sigmoidData * (1 - sigmoidData)
	
	#make the augmented pattern vector
	augPatterns = np.ones((len(patterns), len(patterns[0]) + 1))
	augPatterns[:, 1:] = patterns
	
	#multiply by the pattern vector
	#is one term of the sum
	sumTerm = np.zeros((len(patterns), len(patterns[0]) + 1))

	#can't find any numpy magic so use a loop
	for i in range(len(coefficient)):
		sumTerm[i] = augPatterns[i] * coefficient[i]

	#sum values
	return np.sum(sumTerm, axis = 0)

epsilon = 0.01

oldWeight = np.array([-5, 0.5, 2])
#print(oldWeight)
#print(meanSquareError(petalData, oldWeight, patterns))

oldDecX, oldDecY = decisionBoundary(oldWeight)

newWeight = oldWeight - epsilon * gradientError(petalData, oldWeight, patterns)

#print(meanSquareError(petalData, newWeight, patterns))
newDecX, newDecY = decisionBoundary(newWeight)

plt.figure(6)
plt.scatter(class2[:,PLENGTH], class2[:,PWIDTH], color='blue', marker="2")
plt.scatter(class3[:,PLENGTH], class3[:,PWIDTH], color='green', marker="+")
plt.xlabel("Petal Length")
plt.ylabel("Petal Width")
plt.plot(oldDecX, oldDecY, label="Original Boundary", color='red')
plt.plot(newDecX, newDecY, label="Updated Boundary", color='purple')
plt.legend()
plt.title("Decision Boundary After One Step of Gradient Descent")

################################################################
#3a)
def nextWeight(alpha, beta, error, gradient, weight, petalData, patterns):
	#backtracking line search:
	stepSize = alpha
	while totSquareError(petalData, weight - (stepSize * gradient), patterns) > error - (0.5 * stepSize * la.norm(gradient) ** 2):
		stepSize = stepSize * beta
	return weight - (stepSize * gradient)

def gradientDescent(initialWeight, petalData, patterns):
	#define constants
	alpha = 1
	beta = 0.5
	tolerance = 0.005

	currentWeight = initialWeight
	improvement = 1
	currentError = totSquareError(petalData, currentWeight, patterns)
	#if misclassifies more than it classifies correctly, reverse it.
	if currentError > 25 and totSquareError(petalData, -1 * currentWeight, patterns) < currentError:
		currentWeight = -1 * currentWeight
		currentError = totSquareError(petalData, currentWeight, patterns)
	currentGrad = gradientError(petalData, currentWeight, patterns)
	
	finalWeight = np.zeros((1,3))
	#set a hard max on interations just in case.
	iteration = 0

	#make storage for learning curve info
	iterationList = [0]
	objectiveFunc = [currentError]
	weights = np.array([currentWeight])
	while la.norm(currentGrad) > tolerance and iteration < 100000:
		#finalWeight will be the most recent weight with an improvement.
		if improvement > 0:
			finalWeight = currentWeight

		#update weight using backtracking search
		currentWeight = nextWeight(alpha, beta, currentError, currentGrad, currentWeight, petalData, patterns)
		
		#find next error 
		nextError = totSquareError(petalData, currentWeight, patterns)
		improvement = currentError - nextError
		currentError = nextError

		#update gradient (for stopping condition)
		currentGrad = gradientError(petalData, currentWeight, patterns)
		iteration = iteration + 1

		#store arrays of values to be returned
		weights = np.append(weights, np.array([currentWeight]), axis=0)
		iterationList = np.append(iterationList, iteration)
		objectiveFunc = np.append(objectiveFunc, currentError)

	#if last step was still an improvement, update final weight. Otherwise want to ignore final weight
	if improvement > 0:
		finalWeight = currentWeight

	return finalWeight, iterationList, objectiveFunc, weights


################################################################
#3b)
#define an initial weight
initialWeight = np.array([-5, 0.5, 2])
#get points for a decision boundary with this weight
initX, initY = decisionBoundary(initialWeight)

finalWeight, learningX, learningY, weightHist = gradientDescent(initialWeight, petalData, patterns)
print("3b: initial error: " + str(totSquareError(petalData, initialWeight, patterns)))
print("3b: final error: " + str(totSquareError(petalData, finalWeight, patterns)))
print("3b: final weight: " + str(finalWeight))
finalX, finalY = decisionBoundary(finalWeight)

plt.figure(7)
plt.subplot(211)
plt.scatter(class2[:,PLENGTH], class2[:,PWIDTH], color='blue', marker="2")
plt.scatter(class3[:,PLENGTH], class3[:,PWIDTH], color='green', marker="+")
plt.xlabel("Petal Length")
plt.ylabel("Petal Width")
plt.plot(initX, initY, label="Original Boundary", color='red')
plt.plot(finalX, finalY, label="Optimal Boundary", color='purple')
plt.ylim((0.8, 2.7))
plt.xlim((2.5, 7))
plt.legend()
plt.title("Decision Boundary Optimized by Gradient Descent")


plt.subplot(212)
plt.plot(learningX, learningY, color = "black")
plt.xlabel("Iteration")
plt.ylabel("Objective Function")
plt.title("Learning curve")

plt.tight_layout()


################################################################
#3c)
#selects a random point in the window.  Selects a random slope. Draws the line through that point.
def randWeightOnPlot():
	initX1 = np.random.uniform(low = 2.5, high = 7.0)
	initX2 = np.random.uniform(low = 0.8, high = 2.7)
	W1 = np.random.uniform(low = -10, high = 10)
	W2 = np.random.uniform(low = -10, high = 10)
	W0 = -1 * W1 * initX1 - W2 * initX2
	return np.array([W0, W1, W2])


initialWeight = randWeightOnPlot()
#initialWeight = np.array([-13.03653353, 3.85674907, -6.76452996])
randX, randY = decisionBoundary(initialWeight)


epsilon = 0.01
tolerance= 0.005
finalWeight, learningX, objectiveFuncs, weightHist = gradientDescent(initialWeight, petalData, patterns)

initError = objectiveFuncs[0]
finalError = totSquareError(petalData, finalWeight, patterns)
errordiff = initError - finalError

#find a middle value, error is 95% to final error
middleIndex = 1
found = False
while middleIndex < len(objectiveFuncs) and not found:
	if objectiveFuncs[middleIndex] < initError - 0.95 * errordiff:
		found = True
	else:
		middleIndex = middleIndex + 1


middleWeight = weightHist[middleIndex]
middleError = totSquareError(petalData, middleWeight, patterns)

print("3c: initial weight " + str(initialWeight))
print("3c: initial Error " + str(initError))
print("3c: intermediate found after iteration: " + str(middleIndex))
print("3c: intermediate weight " + str(middleWeight))
print("3c: intermediate error " + str(middleError))
print("3c: convergence after iteration: " + str(len(learningX)))
print("3c: final weight " + str(finalWeight))
print("3c: final error " + str(finalError))

middleX, middleY = decisionBoundary(middleWeight)
finalX, finalY = decisionBoundary(finalWeight)


plt.figure(8)
plt.subplot(211)
plt.scatter(class2[:,PLENGTH], class2[:,PWIDTH], color='blue', marker="2")
plt.scatter(class3[:,PLENGTH], class3[:,PWIDTH], color='green', marker="+")
plt.xlabel("Petal Length")
plt.ylabel("Petal Width")
plt.plot(randX, randY, label="Original Boundary", color='red')
plt.plot(middleX, middleY, label="Middle Boundary", color='orange')
plt.plot(finalX, finalY, label="Optimal Boundary", color='purple')
plt.ylim((0.8, 2.7))
plt.xlim((2.5, 7))
plt.legend()
plt.title("Decision Boundary Optimized by Gradient Descent")

plt.subplot(212)
plt.plot(learningX, objectiveFuncs, label="Learning curve", color = "black")
plt.plot(np.array([middleIndex, middleIndex]), np.array([1, initError]), label="Intermediate Weight", color= "orange")
plt.xlabel("Iteration")
plt.ylabel("Objective Function")
plt.title("Learning curve")
plt.legend()

plt.tight_layout()
################################################################
# show plots at end because it's most convenient
plt.show()
#[ 0.31434082  1.05871523 -3.42272521]
