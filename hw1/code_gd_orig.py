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

def check_MSE(X, Y, W):
	Y2 = matrix(X) * matrix(convert_matrix(W))
	Diff = Y2 - matrix(Y)
	mse = 0
	for value in Diff:
	  mse += value ** 2 / len(Diff)
	return mse[0,0]

def H(W,X):
	score = 0
	# for all features of row X[t]
	for indx in range(0, len(W)):
		score += W[indx] * X[indx]
	#print 'score = ',score
	return score 


#X = array(X)
#Y = array(Y)
def batch(X,Y):
	lmbd = 0.0000148045
	W = [0] * len(X[0])
	# t = 1 to m
	for j in range(0, len(X[0])):
		sigma = 0.0
		for t in range(0, len(X)):
			#print "y is - ", Y[t]
			sigma +=  (H(W, X[t]) - Y[t][0]) * X[t][j]
		print sigma
		W[j] = W[j] - lmbd * sigma
	return W

def stochastic(X,Y):
	best_W = []
	best_mse = 99999999
	lmbd = 0.0001
	W = [0] * len(X[0])
	# t = 1 to m
	for t in range(0, len(X)):
		sigma = 0.0
		for j in range(0, len(X[0])):
			score = (H(W, X[t]) - Y[t][0]) * X[t][j]
			if abs(score) > 1000:
				continue
			cur_mse = check_MSE(X, Y, W)
			if cur_mse < best_mse:
				best_W = list(W)
				best_mse = cur_mse
			#print "y is - ", Y[t]			
			#print score			
			W[j] = W[j] - lmbd * score
			#print W
	return best_W, best_mse

def convert_matrix(mat):
	lst = []
	for val in mat:
		lst.append([val])
	return lst

X,Y = readfile("training")
W, MSE = stochastic(X,Y)

XT, YT = readfile("test")
print "Training score - ", MSE
print "Testing score - ", check_MSE(XT, YT, W)