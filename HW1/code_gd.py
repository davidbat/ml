from numpy import *
def normList(L, normalizeTo=1):
  '''normalize values of a list to make its max = normalizeTo'''
  vMax = max(L)
  return [ x/(vMax*1.0)*normalizeTo for x in L]

def readfile(filename):
  fd = open(filename)
  features = []
  label = []
  for line in fd.readlines(): 
    line = line.strip().split()
    #features.append(normList(map(lambda x:float(x),line[0:-1])))
    features.append(map(lambda x:float(x),line[0:-1]))
    label.append([float(line[-1])])
  for index in range(len(features[0])):
    col = normList(map(lambda row:row[index], features))
    for f in range(len(features)):
      features[f][index] = col[f]
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
  err = 0.001
  W = [0] * len(X[0])
  prevMSE = check_MSE(X, Y, W)
  cond = True
  # t = 1 to m
  cnt = 0
  while cond:
    cnt+= 1
    for t in range(0, len(X)):
      for j in range(0, len(X[0])):
        score = (H(W, X[t]) - Y[t][0]) * X[t][j]
        W[j] = W[j] - lmbd * score
    newMSE = check_MSE(X, Y, W)
    cond = (abs(newMSE - prevMSE) > err)
    prevMSE = newMSE
    if cnt > 1500:
      cond = False
  print cnt
  return W

def convert_matrix(mat):
  lst = []
  for val in mat:
    lst.append([val])
  return lst

X,Y = readfile("training")
W = stochastic(X,Y)

XT, YT = readfile("test")
print "Training score - ", check_MSE(X, Y, W)/2
print "Testing score - ", check_MSE(XT, YT, W)/2