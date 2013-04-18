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

W_I_H = ReadFile("WIH")
W_H_O = ReadFile("WHO")
BIAS_H = map(lambda r:float(r.strip()), open("BH").readlines())
BIAS_O = map(lambda r:float(r.strip()), open("BO").readlines())

jump = 10
if len(sys.argv) > 1:
	jump = int(sys.argv[1])

tstip = ReadFile_hash("./data/ann/test_s_ip.txt", jump)
tstop = ReadFile("./data/ann/test_s_op.txt", jump)
ecoc = open("./data/ann/ecoc.txt").readlines()
ec = {}
for line in ecoc:
  line = line.strip().split()
  ec[float(line[0])] = map(lambda r:float(r), line[1:])

#print ec
temp_arr = []
err = 0
for t in range(len(tstip)):
	Hidden_Out = calculate_sig_hash_list(tstip[t], W_I_H, BIAS_H) 
	#print " ".join(["%.2f" % val for val in Hidden_Out]), "\t\t\t\t\t", " ".join([str(int(round(val))) for val in Hidden_Out])
	Output_Out = map(lambda row:round(row), calculate_sig(Hidden_Out, W_H_O, BIAS_O))

	#temp_arr.append(neatest(Output_Out))
	pred, min_dist = nearest(Output_Out)
	actual = tstop[t]
	#print Output_Out, ec[int(actual[0])], pred, actual[0], min_dist
	#print pred-1, actual[0]-1
	err += 0 if pred == actual[0] else 1

print "%err over", len(tstip), "test data points = ", float(err) / len(tstip) * 100