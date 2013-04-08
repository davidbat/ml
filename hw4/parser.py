import quadprog
from cvxopt.solvers import qp, options
from cvxopt import matrix
import numpy

input_fn = "./data/svmLightTrainingData.txt"

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


def dot_product(hash1, hash2):
	''' xi * xj * yi * yj'''
	return hash1['y'] * hash2['y'] * hash_dot_product(hash1, hash2)


def hash_mult_optimized(hashes):
	length = len(hashes)
	full_mat = [[0 for i in range(length)] for j in range(length)]
	for i in range(length):
		#print i
		for j in range(i,length):
			tmp_val = dot_product(hashes[i],hashes[j])
			#if i == j:
				#print i, hashes[i]
				#print j, hashes[j]
				#print "VAL", tmp_val
			full_mat[i][j] = tmp_val
			full_mat[j][i] = tmp_val
	return full_mat


def calculate_w(hashes, alphas):
	for data_point in hashes:
		sum(map(lambda k1:h1[k1] * h2[k1] if k1 in h2 else 0, data_point))


def one_vs_all(label, jumps, num_features=784):
	hashes = ReadFile_hash(input_fn, num_features, label, jumps)
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
	print alphas


alphas = one_vs_all(1, 300)

