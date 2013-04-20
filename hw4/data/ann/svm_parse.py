ecoc = open("ecoc_svm.txt").readlines()
ec = {}
itrs = 0
for line in ecoc:
  line = line.strip().split()
  itrs = len(line) - 1
  ec[line[0]] = line[1:]
fd=open("../svmLightTrainingData.txt")
#fo=open("train_s_op.txt","w")
#fi=open("train_s_ip.txt","w")
fds = []
for i in range(itrs):
  fds.append(open("./svm/train_"+str(i), "w"))
cnt = -1
for lines in fd.readlines():
  cnt += 1
  if cnt % 10 != 0:
    continue
  lines = lines.strip().split()
  if ec[lines[0]] == []:
    print "here"
  for i in range(itrs):
    fds[i].write(ec[lines[0]][i] + " " + " ".join(lines[1:]) + "\n")
  #fo.write(" ".join(ec[lines[0]]) + "\n")
  #fi.write(" ".join(lines[1:]) + "\n")
fd.close()
for fd in fds:
  fd.close()

