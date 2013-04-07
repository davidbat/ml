#!/usr/bin/python
import random
import math
import operator

fn = "spamset"
k = 10
prior_spam = 0.2

def normList(L, normalizeTo=1):
  '''normalize values of a list to make its max = normalizeTo'''
  vMax = max(L)
  return [ x/(vMax*1.0)*normalizeTo for x in L]

def ReadFile(fn):
  features = []
  for line in open(fn).readlines():
    features.append(map(lambda i: float(i), line.replace(',',' ').split()))
  # dont normalize because gaussian gets worse
  #for index in range(len(features[0]) - 1):
  #  col = normList(map(lambda row:row[index], features))
  #  for f in range(len(features)):
  #    features[f][index] = col[f]
  return features

def calculate_mean(db, st, end):
  means = []
  for index in range(st, end):
    means.append(float(sum(map(lambda row:row[index], db))) / len(db))
  return means

def calculate_bins(db, st, end):
  bins = []
  spam = [item for item in db if item[-1] == 1]
  nspam = [item for item in db if item[-1] == 0]
  for index in range(st, end):
    mean = float(sum(map(lambda row:row[index], db))) / len(db)
    spam_mean = float(sum(map(lambda row:row[index], spam))) / len(spam)
    nspam_mean = float(sum(map(lambda row:row[index], nspam))) / len(nspam)
    mini = float(min(map(lambda row:row[index], db)))
    maxi = float(max(map(lambda row:row[index], db)))
    bins.append(sorted([mini, maxi, mean, spam_mean, nspam_mean]))
  return bins

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

def auc(x,y):
  s = 0.0 

  for i in range(1, len(x)):
    #print x[i] - x[i-1],y[i] + y[i-1]
    s += (x[i] - x[i-1]) * (y[i] + y[i-1])
  return s / 2.0
  
# 1   Pr[fi <= mui | spam]
# 2   Pr[fi > mui | spam]
# 3   Pr[fi <= mui | non-spam]
# 4   Pr[fi > mui | non-spam] 

def bernoulli(db, means, test):
  bern_db = []
  #for each in db:
  # bern_db.append(map(lambda val, mean:int(val > mean) , each, attributes))
  # for every feature
  num_features = len(db[0]) - 1
  num_spam = sum(map(lambda val:val[-1], db))
  num_nspam = len(db) - num_spam
  alph = 1.0 / num_features
  d = num_features
  # for every feature
  for i in range(len(db[0]) - 1):
    inner = []
    p_less_spam = (map(lambda val:int((val[i] <= means[i]) & (val[-1] == 1)), db).count(1) + alph) / (num_spam + alph * d)
    p_more_spam = 1 - p_less_spam
    #p_more_spam = float(map(lambda val:int((val[i] > means[i]) & (val[-1] == 1)), db).count(1) + alph) / (num_spam + alph * d)
    p_less_nspm = (map(lambda val:int((val[i] <= means[i]) & (val[-1] == 0)), db).count(1) + alph) / (num_nspam + alph * d)
    p_more_nspm = 1 - p_less_nspm
    #p_more_nspm = (map(lambda val:int((val[i] > means[i]) & (val[-1] == 0)), db).count(1) + alph) / (num_nspam + alph * d)
    inner = [p_less_spam, p_more_spam, p_less_nspm, p_more_nspm]
    #if p_less_spam ==0 or p_more_spam ==0 or p_less_nspm == 0 or p_more_nspm == 0 or p_less_spam+p_more_spam <= .99 or p_less_spam+p_more_spam >= 1.01:
    #  print inner
    #print inner
    bern_db.append(inner)
  #print bern_db

  FP = 0
  FN = 0
  P = float(map(lambda row:row[-1], test).count(1))
  N = len(test) - P
  TP = 0
  TN = 0
  tau = []
  for row in test:
    spam = math.log(prior_spam, 2)
    nspam = math.log((1 - prior_spam), 2)
    for i in range(len(test[0]) - 1):
      #if bern_db[i][0] * bern_db[i][1] * bern_db[i][2] * bern_db[i][3] == 0:
      # print bern_db[i]
      if row[i] <= means[i]:
        spam += math.log(bern_db[i][0], 2)
        nspam += math.log(bern_db[i][2], 2)
      else:
        spam += math.log(bern_db[i][1], 2)
        nspam += math.log(bern_db[i][3], 2)
    tau.append([spam - nspam, row[-1]])
    if spam > nspam:
      # prediction is spam but it actually isn't then FP
      if row[-1] != 1:
        FP += 1.0
      else:
        TP += 1.0
    else:
      # if predicted as not spam when it is then FN
      if row[-1] != 0:
        FN += 1.0
      else:
        TN += 1.0
  #first cutoff everything is not spam
  ROC_TPR = []
  ROC_FPR = []
  tau = sorted(tau, key=lambda row:row[0], reverse = True)
  for i in range(len(tau) + 1):
    TPR = map(lambda row:row[1],tau[0:i]).count(1) / P
    FPR = map(lambda row:row[1],tau[0:i]).count(0) / N
    ROC_TPR.append(TPR)
    ROC_FPR.append(FPR)
  AUC = auc(ROC_FPR, ROC_TPR)
  return [FP, FN, 1 - ((TP + TN) / (P + N)), ROC_TPR, ROC_FPR, AUC]
      

def Histo(db, bins, test):
  bern_db = []
  #for each in db:
  # bern_db.append(map(lambda val, mean:int(val > mean) , each, attributes))
  # for every feature
  num_features = len(db[0]) - 1
  num_spam = sum(map(lambda val:val[-1], db))
  num_nspam = len(db) - num_spam
  alph = 1.0 / num_features
  d = num_features
  for i in range(len(db[0]) - 1):
    inner = []
    sb1 = (map(lambda val:int((val[i] < bins[i][1]) & (val[-1] == 1)), db).count(1) + alph) / (num_spam + alph * d)
    sb2 = (map(lambda val:int((val[i] >= bins[i][1]) & (val[i] < bins[i][2]) & (val[-1] == 1)), db).count(1) + alph) / (num_spam + alph * d)
    sb3 = (map(lambda val:int((val[i] >= bins[i][2]) & (val[i] < bins[i][3]) & (val[-1] == 1)), db).count(1) + alph) / (num_spam + alph * d)
    sb4 = (map(lambda val:int((val[i] >= bins[i][3]) & (val[-1] == 1)), db).count(1) + alph) / (num_spam + alph * d)
    
    nsb1 = (map(lambda val:int((val[i] < bins[i][1]) & (val[-1] == 0)), db).count(1) + alph) / (num_nspam + alph * d)
    nsb2 = (map(lambda val:int((val[i] >= bins[i][1]) & (val[i] < bins[i][2]) & (val[-1] == 0)), db).count(1) + alph) / (num_nspam + alph * d)
    nsb3 = (map(lambda val:int((val[i] >= bins[i][2]) & (val[i] < bins[i][3]) & (val[-1] == 0)), db).count(1) + alph) / (num_nspam + alph * d)
    nsb4 = (map(lambda val:int((val[i] >= bins[i][3]) & (val[-1] == 0)), db).count(1) + alph) / (num_nspam + alph * d)

    inner = [sb1, sb2, sb3, sb4, nsb1, nsb2, nsb3, nsb4]
    bern_db.append(inner)
  #print bern_db

  FP = 0
  FN = 0
  P = float(map(lambda row:row[-1], test).count(1))
  N = len(test) - P
  TP = 0
  TN = 0
  tau = []
  for row in test:
    spam = math.log(prior_spam, 2)
    nspam = math.log((1 - prior_spam), 2)
    for i in range(len(test[0]) - 1):
      #if bern_db[i][0] * bern_db[i][1] * bern_db[i][2] * bern_db[i][3] == 0:
      # print bern_db[i]
      if row[i] < bins[i][1]:
        spam += math.log(bern_db[i][0], 2)
        nspam += math.log(bern_db[i][4], 2)
      elif row[i] < bins[i][2]:
        spam += math.log(bern_db[i][1], 2)
        nspam += math.log(bern_db[i][5], 2)
      elif row[i] < bins[i][3]:
        spam += math.log(bern_db[i][2], 2)
        nspam += math.log(bern_db[i][6], 2)
      else:
        spam += math.log(bern_db[i][3], 2)
        nspam += math.log(bern_db[i][7], 2)
    tau.append([spam - nspam, row[-1]])
    if spam > nspam:
      # prediction is spam but it actually isn't then FP
      if row[-1] != 1:
        FP += 1.0
      else:
        TP += 1.0
    else:
      # if predicted as not spam when it is then FN
      if row[-1] != 0:
        FN += 1.0
      else:
        TN += 1.0
  #first cutoff everything is not spam
  ROC_TPR = []
  ROC_FPR = []
  tau = sorted(tau, key=lambda row:row[0], reverse = True)
  for i in range(len(tau) + 1):
    TPR = map(lambda row:row[1],tau[0:i]).count(1) / P
    FPR = map(lambda row:row[1],tau[0:i]).count(0) / N
    ROC_TPR.append(TPR)
    ROC_FPR.append(FPR)
  AUC = auc(ROC_FPR, ROC_TPR)
  return [FP, FN, 1 - ((TP + TN) / (P + N)), ROC_TPR, ROC_FPR, AUC]  

def make_gauss(N, mu, sigma):
  k = N / math.sqrt(2*sigma*math.pi)
  s = -1.0 / (2 * sigma)
  def f(x):
    #print "k x mu sigma-", k, x, mu, sigma
    #print "val ", s * (x - mu)*(x - mu)
    #print "other-", math.exp(s * (x - mu)*(x - mu))
    return math.log(k) + (s * (x - mu)*(x - mu))
  return f

def gaussian(train_set, test):
  guass_fncts = []
  train_set_0 = [items for items in train_set if items[-1] == 0]
  train_set_1 = [items for items in train_set if items[-1] == 1]
  num_features = len(train_set[0]) - 1
  for i in range(num_features):
    mu0 = (1.0 / len(train_set_0)) * sum(map(lambda row:row[i], train_set_0))
    mu1 = (1.0 / len(train_set_1)) * sum(map(lambda row:row[i], train_set_1))
    sigma0 = (1.0 / (len(train_set_0) + num_features)) * (sum(map(lambda row:(row[i] - mu0) ** 2, train_set_0)) + 1)
    sigma1 = (1.0 / (len(train_set_1) + num_features)) * (sum(map(lambda row:(row[i] - mu1) ** 2, train_set_1)) + 1)
    guass_fncts.append([make_gauss(1.0, mu0, sigma0), make_gauss(1.0, mu1, sigma1)])

  FP = 0
  FN = 0
  P = float(map(lambda row:row[-1], test).count(1))
  N = len(test) - P
  TP = 0
  TN = 0
  tau = []
  for row in test:
    spam = math.log(prior_spam)
    nspam = math.log((1 - prior_spam))
    for i in range(len(row) - 1):
      spam += guass_fncts[i][1](row[i])
      nspam += guass_fncts[i][0](row[i])
    tau.append([spam - nspam, row[-1]])
    if spam > nspam:
      # prediction is spam but it actually isn't then FP
      if row[-1] != 1:
        FP += 1.0
      else:
        TP += 1.0
    else:
      # if predicted as not spam when it is then FN
      if row[-1] != 0:
        FN += 1.0
      else:
        TN += 1.0
  #first cutoff everything is not spam
  ROC_TPR = []
  ROC_FPR = []
  tau = sorted(tau, key=lambda row:row[0], reverse = True)
  for i in range(len(tau) + 1):
    TPR = map(lambda row:row[1],tau[0:i]).count(1) / P
    FPR = map(lambda row:row[1],tau[0:i]).count(0) / N
    ROC_TPR.append(TPR)
    ROC_FPR.append(FPR)
  AUC = auc(ROC_FPR, ROC_TPR)
  return [FP, FN, 1 - ((TP + TN) / (P + N)), ROC_TPR, ROC_FPR, AUC]    

def pr(arr):
  print "FP\tFN\tErr\tAUC"
  for row in arr:
    print row[0],"\t", row[1], "\t%.2f" % (row[2] * 100), "\t%.6f" % row[5]

def apply_k_fold(folds, means, bins):
  test_index = len(folds) - 1
  length = len(folds)
  bern = []
  gauss = []
  histo = []
  first = 0
  while test_index >= 0:
    #print "test run - ",test_index
    train_set = [item for sublist in folds[0:test_index]+folds[test_index+1:length] for item in sublist]
    test_set = folds[test_index]
    #print len(train_set) , len(test_set)
    #print "*********************TRAINING SET -"
    #print train_set
    #print "*********************TESTING SET -"
    #print test_set
    bern.append(bernoulli(train_set, means, test_set))
    histo.append(Histo(train_set, bins, test_set))
    gauss.append(gaussian(train_set, test_set))
    test_index -= 1
  print "\nBern -"
  pr(bern)
  print "avg error - %.2f" % (sum(map(lambda row:row[2], bern)) / float(len(bern)) *100)
  open("B_T","w").write(" ".join(map(lambda row:str(row), bern[0][3])))
  open("B_F","w").write(" ".join(map(lambda row:str(row), bern[0][4])))
  print "\nGauss -"
  pr(gauss)
  print "avg error - %.2f" % (sum(map(lambda row:row[2], gauss)) / float(len(gauss)) *100)
  open("G_T","w").write(" ".join(map(lambda row:str(row), gauss[0][3])))
  open("G_F","w").write(" ".join(map(lambda row:str(row), gauss[0][4])))
  print "\nHistogram -"
  pr(histo)
  print "avg error - %.2f" % (sum(map(lambda row:row[2], histo)) / float(len(histo)) *100)
  open("H_T","w").write(" ".join(map(lambda row:str(row), histo[0][3])))
  open("H_F","w").write(" ".join(map(lambda row:str(row), histo[0][4])))

dataset = ReadFile("spamset")
#attributes = ReadFile("attributes")
#means = map(lambda row:row[3], attributes)
means = calculate_mean(dataset, 0, len(dataset[0]) - 1)
bins = calculate_bins(dataset, 0, len(dataset[0]) - 1)
#std_dev = map(lambda row:row[3], attributes)
folds = k_fold(dataset, 10)
apply_k_fold(folds, means, bins)