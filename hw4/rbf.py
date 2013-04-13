import quadprog
from cvxopt.solvers import qp, options
from cvxopt import matrix
import numpy
from copy import deepcopy
import sys
import math
#input_fn = "./data/1and2.txt"

def ReadFile(fn, jumps, num_features=784):
	print "Reading file - ", fn
	fd = open(fn)
	lines = fd.readlines()
	data = []
	i = -1
	for line in lines:
		i += 1
		if i % jumps != 0:
			continue
		line = line.split()
		tmp = [ 0.0 for i in range(num_features + 1) ]
		for item in line[1:]:
			item = item.split(":")
			tmp[int(item[0])] = float(item[1])
		tmp[-1] = float(line[0])
		data.append(tmp)
	return data

def ReadFile_hash(fn, jumps):
	print "Reading file - ", fn
	fd = open(fn)
	lines = fd.readlines()
	hashes = []
	i = -1
	for line in lines:
		i += 1
		if i % jumps != 0:
			continue
		
		#print i
		tmp_hash = {}
		line = line.split()
		tmp_hash['y'] = float(line[0])
		for item in line [1:]:
			item = item.split(":")
			# keys are ints and values are floats
			tmp_hash[int(item[0])] = float(item[1])
		hashes.append(tmp_hash)
	print "Finished reading the data"
	return hashes


def euc_product_hash_hash(mh1, mh2):
	''' xi * xj * yi * yj'''
	# We minus 1 here because we do h[]
	h1_keys = set(mh1) - set(['y'])
	only_h2_keys = set(mh2) - h1_keys - set(['y'])
	a = sum(map(lambda k1:(mh1[k1] - mh2[k1])**2 if k1 in mh2 else mh1[k1]**2, h1_keys))
	b = sum(map(lambda k2:mh2[k2]**2, only_h2_keys))
	#print a+b
	return a+b

def rbf_kernel(hashes, g):
	length = len(hashes)
	#print length
	full_mat = [[0.0 for i in range(length)] for j in range(length)]
	for i in range(length):
		#print i
		for j in range(i,length):
			tmp_val = math.exp(-g * euc_product_hash_hash(hashes[i],hashes[j]))
			#print tmp_val
			full_mat[i][j] = tmp_val
			full_mat[j][i] = tmp_val
	return full_mat


def dot_prod_with_y(dot_prod_x, hashes):
	y = [ item['y'] for item in hashes ]
	length = len(hashes)
	for i in range(length):
		#print i
		for j in range(i,length):
			tmp_val = dot_prod_x[i][j] * y[i] * y[j]
			dot_prod_x[i][j] = tmp_val
			dot_prod_x[j][i] = tmp_val
	return dot_prod_x

def one_vs_all(hashes, dot_prod_x):	
	print "Computing dot_prod_matrix"
	full_mat = dot_prod_with_y(dot_prod_x, hashes)
	#print full_mat

	P = matrix(full_mat)
	q = matrix([-1.0 for item in range(len(hashes))])
	tmp_A = map(lambda r:r['y'] , hashes)
	#print tmp_A
	A = matrix(numpy.matrix(tmp_A))
	print "Calculating alphas"
	b = matrix([ 0.0 ])#for i in range(len(hashes))])

	#print qp(P, q, None, None, A, 0.0)

	# H, f, Aeq, beq, lb, ub
	alphas = quadprog.quadprog(full_mat, [-1.0 for item in range(len(hashes))],\
		A, b, matrix([0.0 for item in range(len(hashes))]), \
		matrix([1.0 for item in range(len(hashes))]))
	return alphas


def apply_rbf_kernel(hashes, test_hashes, alphas, g, cur_label, b=0.0):
	summation = 0.0
	non_zero = 0.0
	for indx in range(len(hashes)):
		#if alphas[indx] > 0.0 or True:
			y = 1.0 if hashes[indx]['y'] == cur_label else -1.0
			tmp_val = math.exp(-g * euc_product_hash_hash(hashes[indx], test_hashes)) * y * alphas[indx]
			#if b != 90.0123123123:
			#	print "Inside poly [", indx,"] - ", tmp_val, y, cur_label, hashes[indx]['y'], test_hashes['y']
			summation += tmp_val
			non_zero += 1
	return summation + b#/ non_zero #+ b

def apply_rbf_kernel_dot(hashes, j_index, dot_prod_x, alphas, cur_label, b=0.0):
	summation = 0.0
	non_zero = 0.0
	for indx in range(len(hashes)):
		#if alphas[indx] > 0.0 or True:
			y = 1.0 if hashes[indx]['y'] == cur_label else -1.0
			tmp_val = dot_prod_x[indx][j_index] * y * alphas[indx]
			#if b != 90.0123123123:
			#	print "Inside poly [", indx,"] - ", tmp_val, y, cur_label, hashes[indx]['y'], test_hashes['y']
			summation += tmp_val
			non_zero += 1
	return summation + b#/ non_zero #+ b

def calculate_err(alphas, g,test_hashes, hashes, b):
	print "Calculating error rate"
	err = [ 0 for item in range(10) ]
	my_index_lower = my_index
	my_index_upper = my_index + 1
	if my_index == 0:
		my_index_lower = 1
		my_index_upper = 11
	for indx in range(len(test_hashes)):
		mx = 0
		if my_index == 0:
			mx = -999999999
		mx_label = -1
		for label in range(my_index_lower, my_index_upper):
			cur_pred = apply_rbf_kernel(hashes,test_hashes[indx], alphas[label], g, label, b[label])
			#print cur_pred, label, test_hashes[indx]['y']
			if cur_pred > mx:
				#if test_hashes[indx]['y'] != 1:
				#	err += 1
				mx = cur_pred
				mx_label = label
			#else:
			#	if test_hashes[indx]['y'] == 1:
			#		err += 1
		print "pred -", mx, mx_label-1, test_hashes[indx]['y']-1
		if my_index == 0 and test_hashes[indx]['y'] != mx_label:
			err[int(test_hashes[indx]['y']) - 1] += 1
		if my_index != 0 and ((test_hashes[indx]['y'] == my_index_lower and mx_label == -1) or\
			 (test_hashes[indx]['y'] != my_index_lower and mx_label != -1)):
			err[int(test_hashes[indx]['y']) - 1] += 1
	print err
	return sum(err)


def calculate_b(labeled_hashes, alphas, dot_prod_x, num_features = 784):
	print "Calculating b"
	summation = 0.0
	non_zero = 0.0
	for indx in range(len(labeled_hashes)):
		#if alphas[indx] > 0.0 and False:
		#	continue
		tmp_val = apply_rbf_kernel_dot(labeled_hashes, indx, dot_prod_x, alphas, 1.0)
		#print "at b", tmp_val, labeled_hashes[indx]['y']
		summation +=  labeled_hashes[indx]['y'] - tmp_val
		non_zero += 1
	#print "b=", summation / non_zero
	return summation / non_zero

def iterator(train_fn, test_fn, jumps, test_jumps):
	jumps = jumps
	num_features = 784
	g = 0.000001
	g = 1.0 / (2.0 * pow(850.0,2))
	hashes = ReadFile_hash(train_fn, jumps)
	alphas = {}
	b = {}
	print "Computing prod of x's"
	dot_prod_x = rbf_kernel(hashes, g)
	my_index_lower = my_index
	my_index_upper = my_index + 1
	if my_index == 0:
		my_index_lower = 1
		my_index_upper = 11
	for label in range(my_index_lower,my_index_upper):
		print "Calculations for label - ", label-1
		labeled_hashes = deepcopy(hashes)
		for item in labeled_hashes:
			item['y'] = 1.0 if item['y'] == label else -1.0

		alphas[label] = one_vs_all(labeled_hashes, deepcopy(dot_prod_x))
		alphas[label] = numpy.squeeze(numpy.asarray(alphas[label]))
		#alphas[label] = map(lambda row:0.0 if row < 0 else row, alphas[label])
		#print alphas[label]
		
		b[label] = calculate_b(labeled_hashes, alphas[label], dot_prod_x)
	test_hashes = ReadFile_hash(test_fn, test_jumps)
	print b
	err = calculate_err(alphas, g, test_hashes, hashes, b)
	print err, " predicted wrong out of", len(test_hashes), ". %Err = ", err/float(len(test_hashes))*100

if __name__ == "__main__":
	train_fn = "./data/svmLightTrainingData.txt"
	test_fn = "./data/svmLightTestingData.txt"
	num_features=784
	jumps = int(sys.argv[1])


	my_index = 0
	test_jumps = 50
	if len(sys.argv) == 3:
		test_jumps = int(sys.argv[2])
	iterator(train_fn, test_fn, jumps, test_jumps)