import quadprog
from cvxopt.solvers import qp, options
from cvxopt import matrix
import numpy


#input_fn = "./data/1and2.txt"

def ReadFile_hash(fn, features, label, jumps):
	fd = open(fn)
	lines = fd.readlines()
	data = features
	hashes = []
	i = -1
	for line in lines:
		i += 1
		if i % jumps != 0:
			continue
		
		#print i
		tmp_hash = {}
		line = line.split()
		tmp_hash['y'] = 1.0 if int(line[0]) == label else -1.0
		for item in line [1:]:
			item = item.split(":")
			# keys are ints and values are floats
			tmp_hash[int(item[0])] = float(item[1])
		hashes.append(tmp_hash)
	print "Finished reading the data"
	return hashes


def hash_dot_product(h1, h2):
	if len(h1) > len(h2):
		h1, h2 = h2, h1
	# We minus 1 here because we do h[]
	return sum(map(lambda k1:h1[k1] * h2[k1] if k1 in h2 else 0, h1)) - h1['y'] * h2['y']


def dot_product_hash(hash1, hash2):
	''' xi * xj * yi * yj'''
	return hash1['y'] * hash2['y'] * hash_dot_product(hash1, hash2)


def hash_mult_optimized(hashes):
	length = len(hashes)
	full_mat = [[0 for i in range(length)] for j in range(length)]
	for i in range(length):
		#print i
		for j in range(i,length):
			tmp_val = dot_product_hash(hashes[i],hashes[j])
			#if i == j:
				#print i, hashes[i]
				#print j, hashes[j]
				#print "VAL", tmp_val
			full_mat[i][j] = tmp_val
			full_mat[j][i] = tmp_val
	return full_mat


def calculate_w(hashes, alphas, num_features=784):
	wt = [ 0 for i in range(num_features) ]

	print "Calculating W"
	for indx in range(len(hashes)):
		data_point = hashes[indx]
		for k in data_point:
			if k == 'y':
				continue 
			#print k, indx, alphas[indx] , data_point['y']
			wt[k] += data_point[k] * alphas[indx] * data_point['y']
	return wt

def dot_prod_list(l1,l2):
	return sum(map(lambda x1,x2: x1 * x2,l1,l2))

def calculate_b(x, y, wt, alphas, num_features = 784):
	summation = 0.0
	non_zero = 0.0
	for indx in range(len(x)):
		if alphas[indx] == 0:
			continue
		summation += - dot_prod_list(wt, x[indx]) + y[indx]
		non_zero += 1
	print non_zero
	return summation / non_zero


def one_vs_all(hashes):	
	print "Computing alphas"
	full_mat = hash_mult_optimized(hashes)
	#print full_mat

	P = matrix(full_mat)
	q = matrix([-1.0 for item in range(len(hashes))])
	tmp_A = map(lambda h:h['y'] , hashes)
	#print tmp_A
	A = matrix(numpy.matrix(tmp_A))
	b = matrix([ 0.0 ])#for i in range(len(hashes))])

	#print qp(P, q, None, None, A, 0.0)

	# H, f, Aeq, beq, lb, ub
	alphas = quadprog.quadprog(full_mat, [-1.0 for item in range(len(hashes))],\
		A, b, matrix([0.0 for item in range(len(hashes))]), \
		matrix([1.0 for item in range(len(hashes))]))
	return alphas

def full_x_y(hashes, num_features=784):
	x = []
	y = []
	for indx in range(len(hashes)):
		data_point = hashes[indx]
		tmp = [ 0 for i in range(num_features) ]
		for k in data_point:
			if k == 'y':
				y.append(data_point['y'])
				continue
			#print k, indx, alphas[indx] , data_point['y']
			tmp[k] = data_point[k]
		x.append(tmp)

	return x,y

def calculate_err(wt,b,label,t_x, t_y):
	err = 0
	for indx in range(len(t_x)):
		if dot_prod_list(wt,t_x[indx]) + b > 0:
			if t_y[indx] == -1:
				err += 1
		else:
			if t_y[indx] == 1:
				err += 1
	return err


if __name__ == "__main__":
	train_fn = "./data/svmLightTrainingData.txt"
	test_fn = "./data/svmLightTestingData.txt"
	num_features=784
	jumps = 10
	test_jumps = 5
	label = 1
	hashes = ReadFile_hash(train_fn, num_features, label, jumps)
	alphas = one_vs_all(hashes)
	alphas = numpy.squeeze(numpy.asarray(alphas))
	x,y = full_x_y(hashes)
	wt = calculate_w(hashes, alphas)
	b = calculate_b(x, y, wt, alphas)
	test_hash = ReadFile_hash(train_fn, num_features, label, test_jumps)
	t_x, t_y = full_x_y(test_hash)
	err = calculate_err(wt,b,label,t_x, t_y)
	print "Out of ", len(t_x), " 0 labels, we predicted ", err, " wrong for ", float(len(t_x))/err, "error rate"