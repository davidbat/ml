#!/usr/bin/python

# Assumption -
# 1. Only one hidden layer
import random
import math
import sys

def ReadFile(fn, jumps=1):
	arr = []
	i = -1
	for line in open(fn).readlines():
		i += 1
		if i % jumps != 0:
			continue
		arr.append(map(lambda x:float(x), line.split()))
	return arr

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
		for item in line [1:]:
			item = item.split(":")
			# keys are ints and values are floats
			tmp_hash[int(item[0])-1] = float(item[1])
		hashes.append(tmp_hash)
	print "Finished reading the data"
	return hashes

def sigmoid(x):
	if x <= -100:
		return 1.0 / 2.6881171418161356e+43
	return (1.0 / (1.0 + math.exp(-x)))

def initializeW(R, C):
	return [[random.uniform(-0.1, 0.1) for x in range(C)] for y in range(R)]
	

def sum_mult(A, B):
	if len(A) != len(B):
		print "Matrix lengths not same. Critical error"
		exit(1)
	return sum(map(lambda i,j:i*j, A, B))	

def hash_sum_mult(A, B):
	summation = 0.0
	for key in A:
		summation += A[key] * B[key]
	return summation

def calculate_sig_hash_list(X, W, BIAS):
	O = []
	# for j in every hidden_node
	for j in range(len(W)):
		s = hash_sum_mult(X, W[j])
		O.append(sigmoid(s + BIAS[j]))
	return O

def calculate_sig(X, W, BIAS):
	O = []
	# for j in every output node
	for j in range(0, len(W)):
		s = sum_mult(X, W[j])
		O.append(sigmoid(s + BIAS[j]))
	return O

def network_error(current_op, expected_op):
	if len(current_op) != len(expected_op):
		print "Output matrix lengths not same. Critical error"
		exit(1)
	Err = []
	for index in range(0, len(current_op)):
	  Err.append(current_op[index] * (1 - current_op[index]) * (expected_op[index] - current_op[index]))
	return Err

def hidden_error(HO, NE, W):
	Hidden_Err = []
	# for every hidden node
	for h in range(0, len(HO)):
		net_err = 0.0
		for k in range(0, len(NE)):
			net_err += W[k][h] * NE[k]
		Hidden_Err.append(HO[h] * (1 - HO[h]) * net_err)
	return Hidden_Err

#stateful function
def update_W(W, Err, X, delta_W, BIAS):
	#print learning_rate
	R = len(W)
	C = len(W[0])
	for r in range(0, R):
		for c in range(0, C):
			W[r][c] += learning_rate * Err[r]	* X[c] + momentum * delta_W[r][c]
		BIAS[r] += learning_rate * Err[r]

def update_W_hash(W, Err, X, delta_W, BIAS):
	R = len(W)
	C = len(W[0])
	# for each hidden node
	for r in range(0, R):
		#for c in range(0, C):
		for c in X.keys():
			#learned_val = 0.0
			#if c in X.keys():
				learned_val = learning_rate * Err[r] * X[c]
				#print learning_rate, Err[r], X[c], learned_val
				W[r][c] += learned_val + momentum * delta_W[r][c]
			#else:
			#	print X.keys(), "here"
			#W[r][c] += learned_val + momentum * delta_W[r][c]
		BIAS[r] += learning_rate * Err[r]

def hamdist(lst1, lst2):
	"""Count the # of differences between equal length strings str1 and str2"""
		
	diffs = 0
	for indx in range(len(lst1)):
		if lst1[indx] != lst2[indx]:
			diffs += 1
	return diffs

def nearest(op):
	dist = 0
	min_key = -1
	min_dist = 9999
	for key in ec:
		if hamdist(op, ec[key]) < min_dist:
			min_key = key
			min_dist = hamdist(op, ec[key])
	return min_key, min_dist

conv_err = 0.001
momentum = 0.5
#output_err = 0.2
hidden_nodes = 400
#learning_rate = 0.3
itr = 0

trip = ReadFile_hash("./data/ann/train_s_ip.txt", 75)
trop = ReadFile("./data/ann/train_s_op.txt", 75)
print len(trip), len(trop)
tstip = ReadFile_hash("./data/ann/test_s_ip.txt", 100)
tstop = ReadFile("./data/ann/test_s_op.txt", 100)
ecoc = open("./data/ann/ecoc.txt").readlines()
ec = {}
for line in ecoc:
  line = line.strip().split()
  ec[line[0]] = line[1:]


feature_num = len(trip[0])
feature_num = 784
output_num = len(trop[0])
print feature_num, output_num
BIAS_H = [random.uniform(-0.1, 0.1) for x in range(hidden_nodes)]
BIAS_O = [random.uniform(-0.1, 0.1) for x in range(output_num)] 
W_I_H = initializeW(hidden_nodes, feature_num) # 3 by 8 i.e. 3 arrays with 8 vals each
W_H_O = initializeW(output_num, hidden_nodes) # 8 by 3 i.e. 8 arrays with 3 vals each
assert len(W_I_H) == hidden_nodes
assert len(W_I_H[0]) == feature_num
assert len(W_H_O) == output_num
assert len(W_H_O[0]) == hidden_nodes

def check_diff(trop, trip, W_I_H, W_H_O, BIAS_H, BIAS_O):
	for t in range(0, len(trip)):
		Hidden_Out = calculate_sig_hash_list(trip[t], W_I_H, BIAS_H)
		Output_Out = calculate_sig(Hidden_Out, W_H_O, BIAS_O)
		for j in range(0, len(Output_Out)):
			if abs(Output_Out[j] - trop[t][j]) > output_err:
				return False
	return True

def mean(lst):
	return float(sum(lst))/len(lst)


def calculate_mse(actual, current):
	return mean(map(lambda i,j:abs(i - j), actual, current))

def my_sub(A, B):
	delta = [[0.0 for x in range(len(A[0]))] for y in range(len(A))]
	for i in range(len(A)):
		for j in range(len(A[0])):
			delta[i][j] = A[i][j] - B[i][j]
	return delta


delta_W_I_H = [[0.0 for x in range(feature_num)] for y in range(hidden_nodes)]
delta_W_H_O = [[0.0 for x in range(hidden_nodes)] for y in range(output_num)]


def write_out(lst, fn):
	fd = open(fn, "w")
	for each in lst:
		if type(each) == type([]):
			fd.write(" ".join(map(lambda row:str(row), each)) + "\n")
		else:
			fd.write(str(each) + "\n")
	fd.close()


cnt = 0
cond = True
prev_mse = 999999999
while cond:
	learning_rate = random.uniform(0,0.005)
	err = 0
	prev_W_I_H = list(W_I_H)
	prev_W_H_O = list(W_H_O)
	# for each data point
	for t in range(0, len(trip)):
		#print "Hidden Out"
		Hidden_Out = calculate_sig_hash_list(trip[t], W_I_H, BIAS_H) 
		#print "HO - ", Hidden_Out
		#print "Output"
		Output_Out = calculate_sig(Hidden_Out, W_H_O, BIAS_O)
		#if cnt % 10000 == 1: 
		#print "err"
		err += calculate_mse(trop[t], Output_Out)
		#print "Net err"
		Net_Err = network_error(Output_Out, trop[t])
		#print Output_Out
		#print trop[t]
		#print "Net_Err - ", Net_Err
		#print "Hid err"
		Hid_Err = hidden_error(Hidden_Out, Net_Err, W_H_O)
		#print "Hid_Err - ", Hid_Err
		#print "update WHO"
		update_W(W_H_O, Net_Err, Hidden_Out, delta_W_H_O, BIAS_O)
		#print "update WIH"
		update_W_hash(W_I_H, Hid_Err, trip[t], delta_W_I_H, BIAS_H)
	print err, cnt
	delta_W_I_H = my_sub(W_I_H, prev_W_I_H)
	delta_W_H_O = my_sub(W_H_O, prev_W_H_O)
	#print "Delta WIH\n", delta_W_I_H

	write_out(W_I_H,"WIH")
 	write_out(W_H_O,"WHO")
 	write_out(BIAS_H,"BH")
 	write_out(BIAS_O,"BO")

	cnt += 1
	#cond = (not check_diff(trop, trip, W_I_H, W_H_O, BIAS_H, BIAS_O)) and (cnt <= 100000)
	cond = (cnt < 50) or err > 1# and (abs(err - prev_mse) > conv_err) 
	#cond = (abs(err - prev_mse) > conv_err)
	#prev_mse = mse
	#if cnt % 10000 == 1 or not cond:
	#	print "Iteration - ", cnt
	
def print_arr(lst):
	for row in range(len(lst)):
		print " ".join(["%.2f" % val for val in lst[row]]), "\t", " ".join([str(int(round(val))) for val in lst[row]])

#print "Hidden Vals -"
temp_arr = []
for t in range(len(tstip)):
	Hidden_Out = calculate_sig_hash_list(tstip[t], W_I_H, BIAS_H) 
	#print " ".join(["%.2f" % val for val in Hidden_Out]), "\t\t\t\t\t", " ".join([str(int(round(val))) for val in Hidden_Out])
	Output_Out = calculate_sig(Hidden_Out, W_H_O, BIAS_O)
	#temp_arr.append(neatest(Output_Out))
	pred, min_dist = nearest(Output_Out)
	actual = tstop[t]
	print Output_Out, pred, actual[0], min_dist

#print "\nOutputs Vals\t\t\t\t\tRounded Outputs"
#print_arr(temp_arr)
