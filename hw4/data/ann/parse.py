ecoc = open("ecoc.txt").readlines()
ec = {}
for line in ecoc:
  line = line.strip().split()
  ec[line[0]] = line[1:]
fd=open("../svmLightTrainingData.txt")
fo=open("train_s_op.txt","w")
fi=open("train_s_ip.txt","w")
f3 = open("train_mult.txt","w")
cnt = -1
for lines in fd.readlines():
  cnt += 1
  if cnt % 10 != 0:
    continue
  lines = lines.strip().split()
  if ec[lines[0]] == []:
    print "here"
  fo.write(" ".join(ec[lines[0]]) + "\n")
  fi.write(" ".join(lines[1:]) + "\n")
  f3.write(" ".join(lines) + "\n")
fo.close()
fi.close()
fd.close()
f3.close()
 
fd=open("../svmLightTestingData.txt")
fo=open("test_s_op.txt","w")
fi=open("test_s_ip.txt","w")
f3=open("test_mult.txt", "w")
for lines in fd.readlines():
  lines = lines.strip().split()
  if ec[lines[0]] == []:
    print "here"
  fo.write(lines[0] + "\n")
  fi.write(" ".join(lines[1:]) + "\n")
  f3.write(" ".join(lines)+"\n")
fi.close()
fd.close()
fo.close()
f3.close()
