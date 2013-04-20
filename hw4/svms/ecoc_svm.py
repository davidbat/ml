import os
learn = "/home/dave/ml/hw4/svms/svmlight/svm_learn"
classify = "/home/dave/ml/hw4/svms/svmlight/svm_classify"

tr_ips = "/home/dave/ml/hw4/data/ann/svm/train_"
tst_ip = "/home/dave/ml/hw4/data/svmLightTestingData.txt"
ecoc = 

ecoc = open("../data/ann/ecoc_svm.txt").readlines()
ec = {}
itrs = 0
for line in ecoc:
  line = line.strip().split()
  itrs = len(line) - 1
  ec[line[0]] = line[1:]

for i in range(15):
	os.system(learn+ " -t 1 -d 2 -m 500 " +tr_ips+str(i))
	os.system("mv "+model+" model_"+str(i))