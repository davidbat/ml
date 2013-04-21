from numpy import *

def readfile(filename):
  fd = open(filename)
  features = []
  label = []
  for line in fd.readlines(): 
    line = line.strip().split()
    features.append(map(lambda x:float(x),line[0:-1]))
    label.append([float(line[-1])])
  return features,label

X,Y = readfile("training")

TX, TY = readfile("test")
#print Y
#'''exit(0)
X = asmatrix(X)
Y = asmatrix(Y)
#print Y
XT = transpose(X)
A = (XT * Y)
W = (linalg.pinv(XT * X)) * XT * Y
Y2 = TX * W
Diff = Y2 - TY
mse = 0
for value in Diff:
  mse += value ** 2 / len(Diff)
print "MSE test = ", mse[0,0]

Y3 = X * W
Diff = Y3 - Y
mse = 0
for value in Diff:
  mse += value ** 2 / len(Diff)
print "MSE training = ", mse[0,0]