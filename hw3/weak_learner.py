
import math
class LeafNode:
	left , right, data = None, None, 0.0
	
	def __init__(self, data):
		# initializes the data members
		self.left = None
		self.right = None
		self.data = data

class PredicateNode:
	left , right, feature_id, value = None, None, -1, -1
	
	def __init__(self, ids, val):
		# initializes the data members
		self.left = None
		self.right = None
		self.feature_id = ids

class StumpNode:
	left_prediction , right_prediction, feature_id, value = None, None, -1, -1
	
	def __init__(self, lp, rp, ids, val):
		# initializes the data members
		self.left_prediction = lp
		self.right_prediction = rp
		self.feature_id = ids
		self.value = val

class Decision_Stump:
	left_prediction , right_prediction, feature_id, value, toggle = None, None, -1, -1, False

	def __init__(self, X, Dt):
		#best_IG = -9999
		best_v = -float("inf")
		best_f = -float("inf")
		most_diff = -float("inf")
		for feature_num in range(0, len(X[0]) - 1):
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

	def calculate_err(X, feature_num, val, Dt):
		err = 0
		for i in range(len(X)):
			# prediction is for y = 1
			if X[i][feature_num] == val:
				if X[i][-1] != 1:
					err += Dt[i]
			else:
				if X[i][-1] != -1:
					err += Dt[i]
		return err

	def uniq_labels(self, Y):
		uniq_Y = {}
		for label in Y:
			if label in uniq_Y.keys():
				uniq_Y[label] += 1
			else:
				uniq_Y[label] = 1
		return uniq_Y
	
	def calculate_ig_ent(self, X, feature_num, val):
		# p * log(1 / p)
		# for x-merging we will need to handle a list in labels spot
		entropy_left = 0
		entropy_right = 0
		parent_entropy = 0
		items_num = len(X)
		X_left = [ item for item in map(lambda row:row if row[feature_num] == val else [], X) if item ]
		items_left = len(X_left)
		X_right = [ item for item in map(lambda row:row if row[feature_num] != val else [], X) if item ]
		items_right = len(X_right)

		Y = self.uniq_labels(map(lambda row:row[-1], X))
		for label in Y.keys():
			parent_entropy += (float(Y[label]) / items_num * math.log(items_num / float(Y[label]), 2))

		Y = self.uniq_labels(map(lambda row:row[len(row)-1], X_left))
		for label in Y.keys():
			entropy_left += float(Y[label]) / items_left * math.log(items_left / float(Y[label]), 2)

		Y = self.uniq_labels(map(lambda row:row[len(row)-1], X_right))
		for label in Y.keys():
			entropy_right += float(Y[label]) / items_right * math.log(items_right / float(Y[label]),2)

		IG = float(items_left) / items_num * (parent_entropy - entropy_left) + items_right /items_num * (parent_entropy - entropy_right)
		return IG

Decision_Stump([[1,2,3],[1,1,3],[1,3,4]])
