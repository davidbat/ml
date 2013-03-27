#from weak_learner import *
import os
import math
import sys
from random import choice
import random
import operator

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
	left_prediction , right_prediction, feature_id, value, toggle = None, None, -1, -1, False

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

def active_learning(X, H, alph, growth):
	err = 0.0
	global_mx = float("inf")
	all_values = []
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
		all_values.append([x, abs(mx)])
	return map(lambda row: row[0], sorted(all_values, key=operator.itemgetter(1)))[0:int(growth)]




pwd = os.getcwd()
# [ vote,tic,nur,monk,cmc,band,bal,agr,crx,car ]
fold_num = 10
random_flag = False
args = sys.argv[1:]

#if sys.argv[1] == "-r":
#	random_flag = True
#	args = sys.argv[2:]
for folder in args:
	path = pwd + "/" + folder + "/"
	#config = path + folder + ".config"
	#fc = open(config)
	data = path + folder + ".data"
	Full = ReadFile(data)

	#config = path + folder + ".config"
	# we know that vote is 2 class
	# totally hacked for vote cause i want a running set
	folds = k_fold(Full, fold_num)
	test_index = len(folds) - 1
	length = len(folds)
	
	uniq_labels = set(map(lambda row: row[-1], Full))
	test_index = 0
	output = {}
	itr = 0
	while test_index >= 0:
		itr += 1
		output[test_index] = {}
		#print "test run - ",test_index
		orig_train_set = [item for sublist in folds[0:test_index]+folds[test_index+1:length] for item in sublist]
		growth_rate = 1.0 * len(orig_train_set) / 100
		#growth_rate = 1
		Test = folds[test_index]
		#print "Fold - ", length - test_index
		for c in [5]:
			orig_orig_X = random_choice(orig_train_set, c) 
			#print len(X)
			#print len(Test)
			for run in [ "random", "best"]:
				train_set = list(orig_train_set)
				orig_X = list(orig_orig_X)
				output[test_index][run] = {}
				m = 0
				acc = 0
				while len(orig_X) < (len(orig_train_set) / 5):# and acc < 88:
					m += 1
					#print len(orig_X), (len(Full) / 2)
					alph = {}
					Dt = {}
					H = {}
					alph = {}
					for label in uniq_labels:
						Dt[label] = initialize_wts(len(orig_X))
						H[label] = []
						alph[label] = []
					#print "here"
					#for t in range(500):
					t = 0
					break_err = -1
					while break_err < (0.499 * len(uniq_labels)) and break_err != 0 and t <= 50:
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
					output[test_index][run][float(len(orig_X)/ float(len(orig_train_set)) * 100.0)] = 100 - acc
					if run == "best":
						best_xs = active_learning(train_set, H, alph, growth_rate)
						for best_x in best_xs:
							cnt_best_x = train_set.count(best_x) - 1
							train_set = [ item for item in train_set if item != best_x ]
							train_set += [list(best_x) for l in range(cnt_best_x)]
							orig_X.append(best_x)
					else:
						random_xs = random.sample(train_set, int(growth_rate))
						for random_x in random_xs:
							cnt_random_x = train_set.count(random_x) - 1
							train_set = [ item for item in train_set if item != random_x ]
							train_set += [list(random_x) for l in range(cnt_random_x)]
							orig_X.append(random_x)

				#print "Misclassified =", misclassed
				#print "for c = ", c, " after", t, "iterations error with test set len = ", len(Test), " is - ", test_err, " acc = %.2f" % acc
				#print "Run - ", run
				#print "size\terr"
				#print max(output.keys()), "\t%.2f" % output[max(output.keys())]
				#for size in sorted(output.keys()):
				#	print size, "\t%.2f" % output[size]

		test_index -= 1
	average = {}
	average['best'] = []
	average['random'] = []
	average['key'] = []
	for fold in output.keys():
		for run in output[fold].keys():
			if average['key'] == []:
				average['key'] = sorted(output[fold][run].keys())
			#print len(output[fold][run])
			sorted_values = map(lambda row:float(row[1])/itr, sorted(output[fold][run].iteritems(), key=operator.itemgetter(0)))
			if average[run] == []:
				average[run] = sorted_values
			else:
				if len(sorted_values) < len(average[run]):
					average[run] = average[run][:-1]
				elif len(sorted_values) > len(average[run]):
					sorted_values = sorted_values[:-1]
				average[run] = map(lambda a,b:a+b, sorted_values, average[run])
	print "%Growth\tRandom\tActive"
	for i in range(len(average['best'])):
		print "%.2f" % average['key'][i], "\t%.2f" % average['random'][i], "\t%.2f" % average['best'][i]




	
