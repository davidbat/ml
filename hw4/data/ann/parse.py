ecoc = open("ecoc.txt").readlines()
ec = {}
for line in ecoc:
  line = line.strip().split()
  ec[line[0]] = line[1:]
fd=open("../svmLightTrainingData.txt")
fo=open("train_s_op.txt","w")
fi=open("train_s_ip.txt","w")
for lines in fd.readlines():
  lines = lines.strip().split()
  if ec[lines[0]] == []:
    print "here"
  fo.write(" ".join(ec[lines[0]]) + "\n")
  fi.write(" ".join(lines[1:]) + "\n")
fo.close()
fi.close()
fd.close()
 
fd=open("../svmLightTestingData.txt")
fo=open("test_s_op.txt","w")
fi=open("test_s_ip.txt","w")
for lines in fd.readlines():
  lines = lines.strip().split()
  if ec[lines[0]] == []:
    print "here"
  fo.write(" ".join(ec[lines[0]]) + "\n")
  fi.write(" ".join(lines[1:]) + "\n")
fi.close()
fd.close()
fo.close()
