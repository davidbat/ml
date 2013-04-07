#!/usr/bin/python

# Assumption -
# 1. Only one hidden layer
import random
import math


mode = "test"
if mode == "test":
	hidden_nodes = 2
	learning_rate = 0.9
else:
	hidden_nodes = 3
	learning_rate = 0.05

def ReadFile(fn):
	arr = []
	for line in open(fn).readlines():
		arr.append(map(lambda x:float(x), line.split()))
	return arr

def sigmoid(x):
	print x
	return 1.0 / (1 + math.exp(-x))

def initializeW(R, C):
	W = [[0] * C] * R
	# array of dimensions R x C
	for row in range(0, R):
		for col in range(0, C):
			W[row][col] = random.uniform(-0.05, 0.05)
	return W

# multiple 2 arrays of same length
# for all i: sum(a[i]*b[i])
def sum_mult(A, B):
	if len(A) != len(B):
		print "Matrix lengths not same. Critical error"
		exit(1)
	return sum(map(lambda i,j:i*j, A, B))	

def calculate_sig(X, W, BIAS):
	O = []
	# for j in every hidden_node
	for j in range(0, len(W)):
		O.append(sigmoid(sum_mult(X, W[j]) + BIAS[j]))
	return O


def network_error(current_op, expected_op):
	if len(current_op) != len(expected_op):
		print "Output matrix lengths not same. Critical error"
		exit(1)
	Err = []
	for index in range(0,len(current_op)):
	  Err.append(current_op[index] * (1 - current_op[index]) * (expected_op[index] - current_op[index]))
	return Err

def hidden_error(HO, NE, W):
	Hidden_Err = []
	for h in range(0, len(HO)):
		net_err = 0
		for k in range(0, len(NE)):
			net_err += W[k][h] * NE[k] 
		Hidden_Err.append(HO[h] * (1 - HO[h]) * net_err)
	return Hidden_Err

#stateful function
def update_W(W, Err, X):
	R = len(W)
	C = len(W[0])
	#print "Here with ", R, " rows and ", C, " cols"
	for r in range(0, R):
		for c in range(0, C):
			W[r][c] += learning_rate * Err[r]	* X[c]
			print learning_rate, " * ", Err[r],	" * ", X[c]

if mode != "test":
	trip = ReadFile("tr_ip.txt")
	print trip
	trop = ReadFile("tr_op.txt")
else:
	trip = ReadFile("ex_ip.txt")
	print trip
	trop = ReadFile("ex_op.txt")

feature_num = len(trip[0])
output_num = len(trop[0])


if mode != "test":
	BIAS_H = [0] * hidden_nodes
	BIAS_O = [0] * output_num
	W_I_H = initializeW(hidden_nodes, feature_num) # 3 by 8 i.e. 3 arrays with 8 vals each
	W_H_O = initializeW(feature_num, hidden_nodes) # 8 by 3 i.e. 8 arrays with 3 vals each
else:
	BIAS_H = [-0.4, 0.2]
	BIAS_O = [0.1]
	W_I_H = [[0.2, 0.4, -0.5], [-0.3, 0.1, 0.2]] # 2 by 3
	W_H_O = [[-0.3, -0.2]]

cnt = 0
cond = True
while cond:
	for t in range(0, len(trip)):
		Hidden_Out = calculate_sig(trip[t], W_I_H, BIAS_H) 
		Output_Out = calculate_sig(Hidden_Out, W_H_O, BIAS_O)
		Net_Err = network_error(Output_Out, trop[t])
		Hid_Err = hidden_error(Hidden_Out, Net_Err, W_H_O)
		update_W(W_H_O, Net_Err, Hidden_Out)
		update_W(W_I_H, Hid_Err, trip[t])
	cnt += 1
	#print cnt , "((((((((((((((((((((((((((((("
	cond = (cnt <= 0)

print "Hidden Vals -"
temp_arr = []
for t in range(0, len(trip)):
	Hidden_Out = calculate_sig(trip[t], W_I_H, BIAS_H) 
	print Hidden_Out
	Output_Out = calculate_sig(Hidden_Out, W_H_O, BIAS_O)
	temp_arr.append(Output_Out)

print "\nOutputs -"
print temp_arr
print "\nHidden Weights -", W_I_H
print "\nOutput Weights -", W_H_O
