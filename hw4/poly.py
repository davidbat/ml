import quadprog
from cvxopt.solvers import qp, options
from cvxopt import matrix
import numpy
from copy import deepcopy
import sys
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

def dot_prod_list(l1,l2):
	return sum(map(lambda x1,x2: x1 * x2,l1,l2))

def dot_product_hash_hash(h1, h2):
	''' xi * xj * yi * yj'''
	mh1 = h1
	mh2 = h2
	if len(h1) > len(h2):
		mh1, mh2 = h2, h1
	# We minus 1 here because we do h[]
	return sum(map(lambda k1:mh1[k1] * mh2[k1] if k1 in mh2 else 0.0, list(set(mh1) - set(['y']))))

def linear_kernel(hashes):
	length = len(hashes)
	#print length
	full_mat = [[0.0 for i in range(length)] for j in range(length)]
	for i in range(length):
		#print i
		for j in range(i,length):
			tmp_val = dot_product_hash_hash(hashes[i],hashes[j])
			full_mat[i][j] = tmp_val
			full_mat[j][i] = tmp_val
	return full_mat


def poly_kernel(hashes, d, c):
	length = len(hashes)
	#print length
	full_mat = [[0.0 for i in range(length)] for j in range(length)]
	for i in range(length):
		#print i
		for j in range(i,length):
			tmp_val = pow(dot_product_hash_hash(hashes[i],hashes[j]) + c, d)
			#print tmp_val
			full_mat[i][j] = tmp_val
			full_mat[j][i] = tmp_val
	return full_mat


def calculate_w(hashes, alphas, num_features=784):
	wt = [ 0.0 for i in range(num_features) ]

	print "Calculating W"
	for indx in range(len(hashes)):
		data_point = hashes[indx]
		for k in data_point:
			if k == 'y':
				continue
			#print k, indx, alphas[indx] , data_point['y']
			wt[k] += data_point[k] * alphas[indx] * data_point['y']
	return wt

def dot_product_list_hash(lst1, hash2):
	summation = 0.0
	for key in hash2:
		if key == 'y':
			continue
		summation += lst1[key] * hash2[key]
	return summation

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


def apply_poly_kernel(hashes, test_hashes, alphas, d, c, cur_label, b=0.0):
	summation = 0.0
	non_zero = 0.0
	for indx in range(len(hashes)):
		if alphas[indx] > 0.0 or True:
			y = 1.0 if hashes[indx]['y'] == cur_label else -1.0
			tmp_val = pow(dot_product_hash_hash(hashes[indx], test_hashes) +c, d) * y * alphas[indx]
			#if b != 90.0123123123:
			#	print "Inside poly [", indx,"] - ", tmp_val, y, cur_label, hashes[indx]['y'], test_hashes['y']
			summation += tmp_val
			non_zero += 1
	return summation + b#/ non_zero #+ b

def apply_poly_kernel_dot(hashes, j_index, dot_prod_x, alphas, d, c, cur_label, b=0.0):
	summation = 0.0
	non_zero = 0.0
	for indx in range(len(hashes)):
		if alphas[indx] > 0.0 or True:
			y = 1.0 if hashes[indx]['y'] == cur_label else -1.0
			tmp_val = dot_prod_x[indx][j_index] * y * alphas[indx]
			#if b != 90.0123123123:
			#	print "Inside poly [", indx,"] - ", tmp_val, y, cur_label, hashes[indx]['y'], test_hashes['y']
			summation += tmp_val
			non_zero += 1
	return summation + b#/ non_zero #+ b

def apply_poly_kernel_vote(hashes, test_hashes, alphas, d, c, cur_label, b=0.0):
	summation = 0.0
	for indx in range(len(hashes)):
		if alphas[indx] < 0.0:
			continue
		y = 1.0 if hashes[indx]['y'] == cur_label else -1.0
		if pow(dot_product_hash_hash(hashes[indx], test_hashes) +c, d) * y * alphas[indx] + b > 0 and\
			y == 1:
			summation += 1
		#if b != 0.0:
		#	print "Inside poly [", indx,"] - ", tmp_val, y, test_hashes['y']
	return summation

def calculate_err(alphas, d, c,test_hashes, hashes, b):
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
			cur_pred = apply_poly_kernel(hashes,test_hashes[indx], alphas[label], d, c, label, b[label])
			#print cur_pred, label, test_hashes[indx]['y']
			if cur_pred > mx:
				#if test_hashes[indx]['y'] != 1:
				#	err += 1
				mx = cur_pred
				mx_label = label
			#else:
			#	if test_hashes[indx]['y'] == 1:
			#		err += 1
		print "pred -", mx, mx_label, test_hashes[indx]['y']
		if my_index == 0 and test_hashes[indx]['y'] != mx_label:
			err[int(test_hashes[indx]['y']) - 1] += 1
		if my_index != 0 and ((test_hashes[indx]['y'] == my_index_lower and mx_label == -1) or\
			 (test_hashes[indx]['y'] != my_index_lower and mx_label != -1)):
			err[int(test_hashes[indx]['y']) - 1] += 1
	print err
	return sum(err)


def calculate_b(labeled_hashes, alphas, dot_prod_x, d, c, num_features = 784):
	print "Calculating b"
	summation = 0.0
	non_zero = 0.0
	for indx in range(len(labeled_hashes)):
		if alphas[indx] > 0.0 and False:
			continue
		tmp_val = apply_poly_kernel_dot(labeled_hashes, indx, dot_prod_x, alphas, d, c, 1.0)
		#print "at b", tmp_val, labeled_hashes[indx]['y']
		summation +=  labeled_hashes[indx]['y'] - tmp_val
		non_zero += 1
	#print "b=", summation / non_zero
	return summation / non_zero

def iterator(train_fn, test_fn, jumps, test_jumps):
	jumps = jumps
	num_features = 784
	c = 2
	d = 2
	hashes = ReadFile_hash(train_fn, jumps)
	alphas = {}
	b = {}
	print "Computing prod of x's"
	dot_prod_x = poly_kernel(hashes, d, c)
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
		
		b[label] = calculate_b(labeled_hashes, alphas[label], dot_prod_x, d, c)
	test_hashes = ReadFile_hash(test_fn, test_jumps)
	print b
	err = calculate_err(alphas, d, c, test_hashes, hashes, b)
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
	'''
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
	'''