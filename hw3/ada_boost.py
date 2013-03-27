#from weak_learner import *
import os
import math
import sys
from random import choice
import random


def ReadFile(fn):
	features = []
	for line in open(fn).readlines():
		features.append(map(lambda i: i, line.split()))
	# dont normalize because gaussian gets worse
	#for index in range(len(features[0]) - 1):
	#  col = normList(map(lambda row:row[index], features))
	#  for f in range(len(features)):
	#    features[f][index] = col[f]
	return features

def initialize_wts(num):
	return  [1.0/num] * num

class Decision_Stump:
	feature_id, value, toggle = -1, -1, False

	def __init__(self, X, Dt):
		#best_IG = -9999
		best_v = -float("inf")
		best_f = -float("inf")
		most_diff = -float("inf")
		# reset labels to 1 and -1
		for feature_num in range(0, len(X[0]) - 1):
			# unique values for X[feature_num]
			for val in set(map(lambda row:row[feature_num], X)):
				#IG = self.calculate_ig_ent(X, feature_num, val)
				#if IG > best_IG:
				#	best_v = val
				#	best_IG = IG
				#	best_f = feature_num
				err = calculate_err(X, feature_num, val, Dt)
				if abs(0.5 - err) > most_diff:
					best_v = val
					best_f = feature_num
					least_err = err
					most_diff = abs(0.5 - err)
		#left = [ item for item in map(lambda row:row[-1] if row[best_f] == best_v else [], X) if item ]
		#right = [ item for item in map(lambda row:row[-1] if row[best_f] != best_v else [], X) if item ]
		# toggle prediction
		if least_err > 0.5:
			self.toggle = True
			#left, right = right, left
		#self.left_prediction = max(set(left), key=left.count)
		#self.right_prediction = max(set(right), key=right.count)
		self.feature_id = best_f
		self.value = best_v

	def pr(self):
		print "featurn num - ", self.feature_id
		print "value - ",self. value
		print "toggle - ", self.toggle

def calculate_err(X, feature_num, val, Dt, toggle = False):
	err = 0.0
	for i in range(len(X)):
		# prediction is for y = 1
		if (X[i][feature_num] == val and not toggle) or (X[i][feature_num] != val and toggle):
			if X[i][-1] != 1:
				err += Dt[i]
		else:
			if X[i][-1] != -1:
				err += Dt[i]
	return err

def predict(H, X):
	y = -1
	if X[H.feature_id] == H.value:
		y = 1
	if H.toggle:
		y *= -1
	return y

def random_choice(L, c):
	index = range(len(L))
	pc = len(L) * c / 100
	X = []
	for i in range(pc):
		current_choice = choice(index)
		X.append(L[current_choice])
		index = [item for item in index if item != current_choice]
	return X

def Test_Data(X, H, alph):
	err = 0.0
	cnt = 0
	for x in X:
		total = {}
		for label in H.keys():
			total[label] = 0.0
			for t in range(len(H[label])):
				total[label] += predict(H[label][t], x) * alph[label][t]
		mx = -float("inf")
		mx_label = ""
		for label in total.keys():
			if total[label] > mx:
				mx_label = label
				mx = total[label]
		if mx < 0:
			cnt += 1
		err += 1 if mx_label != x[-1] else 0
			#y = 1 if total >= 0 else -1
			#err += 1 if y != x[-1] else 0
	return err, cnt

def new_fold(db, size):
	#usable = [item for item in db if item not in used]
	length = len(db)
	if length <= size:
		return db
	fold = []
	for i in range(size):
		cur_len = len(db)
		curr_index = random.randint(0,cur_len-1)
		curr_item = db[curr_index]
		db = db[0:curr_index] + db[curr_index+1:cur_len]
		fold.append(curr_item)
	return fold, db

def k_fold(orig_arr, k):
	folds = []
	used = []
	arr = list(orig_arr)
	fold_size = int(round(len(arr) / float(k)))
	for i in range(k-1):
		fold, arr = new_fold(arr, fold_size)
		folds.append(fold)
	# last fold has everything else
	folds.append(arr)
	return folds

pwd = os.getcwd()
# [ vote,tic,nur,monk,cmc,band,bal,agr,crx,car ]
fol_num = 10
for folder in sys.argv[1:]:
	path = pwd + "/" + folder + "/"
	#config = path + folder + ".config"
	#fc = open(config)
	data = path + folder + ".data.bayes"
	Full = ReadFile(data)

	#config = path + folder + ".config"
	# we know that vote is 2 class
	# totally hacked for vote cause i want a running set
	folds = k_fold(Full, fol_num)
	test_index = len(folds) - 1
	length = len(folds)
	
	uniq_labels = set(map(lambda row: row[-1], Full))
	output = {}
	while test_index >= 0:
		#print "test run - ",test_index
		train_set = [item for sublist in folds[0:test_index]+folds[test_index+1:length] for item in sublist]
		Test = folds[test_index]
		#print "Fold - ", length - test_index

		for c in [5, 10, 15, 20, 30, 50, 80]:
			orig_X = random_choice(train_set, c) 
			#print len(X)
			#print len(Test)
			#print c
			Dt = {}
			H = {}
			alph = {}
			for label in uniq_labels:
				Dt[label] = initialize_wts(len(orig_X))
				H[label] = []
				alph[label] = []


			#for t in range(500):
			t = 0
			break_err = -1
			while break_err < (0.499 * len(uniq_labels)) and break_err != 0 and t <= 100:
				t += 1
				break_err = 0.0
				#print t
				#print Dt
				for label in uniq_labels:
					X = map(lambda row:row[:-1] + [1 if row[-1] == label else -1], orig_X)
					cur_stump = Decision_Stump(X, Dt[label])
					#cur_stump.pr()
					err = calculate_err(X, cur_stump.feature_id, cur_stump.value, Dt[label], cur_stump.toggle)
					break_err += err
					#print err
					if err == 0:
						a = 999
					else:
						a = 1.0 / 2 * (math.log((1.0-err)/err))
					alph[label].append(a)
					H[label].append(cur_stump)
					Zt = 2.0 * math.sqrt(err * (1 - err))
					next_Dt = []
					if Zt != 0:
						for i in range(len(Dt[label])):
							next_Dt.append((Dt[label][i] / Zt) * math.exp(-a * X[i][-1] * predict(cur_stump, X[i])))
						Dt[label] = next_Dt

			test_err, misclassed = Test_Data(Test, H, alph)
			acc = 100 - test_err*100.0/len(Test) 
			if output.has_key(c):
				if output[c]['maxi'] < acc:
					output[c]['maxi'] = acc
				output[c]['total'] += acc
			else:
				output[c] = {}
				output[c]['maxi'] = acc
				output[c]['total'] = acc 
			#print "Misclassified =", misclassed
			#print "for c = ", c, " after", t, "iterations error with test set len = ", len(Test), " is - ", test_err, " acc = %.2f" % acc

		test_index -= 1

	print "c\taverage\tmax"
	for c in sorted(output.keys()):
		print c, "\t%.2f" % (output[c]['total'] / fol_num), "\t%.2f" % output[c]['maxi']

