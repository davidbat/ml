import heapq
import os
import sys

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

def median(mylist):
	sorts = sorted(mylist)
	length = len(sorts)
	return sorts[length / 2]

def most_common_or_med(lst):
	maxi, maxi_2 = heapq.nlargest(2, lst)
	if maxi == maxi_2:
		return median(lst)
	else:
		return maxi

def create_missing_hash(data):
	uniq_labels = set(map(lambda row: row[-1], Full))
	missing_hash = {}
	for feature in range(len(data[0])):
		missing_hash[feature] = {}
		for label in uniq_labels:
			features_for_label = map(lambda row: row[feature], filter(lambda x:x[-1] == label, data))
			missing_hash[feature][label] = most_common_or_med(features_for_label)
	return missing_hash

def replace_missing(data, mhash):
	for row in range(len(data)):
		for col in range(len(data[row])):
			if data[row][col] == "?":
				#print col, data[row][-1]
				data[row][col] = mhash[col][data[row][-1]]
	return data

def write_out(lst, fn):
	fd = open(fn+".bayes", "w")
	for row in lst:
		fd.write("\t".join(row) + "\n")
	fd.close()

pwd = os.getcwd()

for folder in sys.argv[1:]:
	path = pwd + "/" + folder + "/"
	#config = path + folder + ".config"
	#fc = open(config)
	data = path + folder + ".data"
	Full = ReadFile(data)
	missing_hash = create_missing_hash(Full)
	Full = replace_missing(Full, missing_hash)
	write_out(Full, data)