ecoc = open("ecoc.txt").readlines()
ec = {}
for line in ecoc:
  line = line.strip().split()
  ec[line[0]] = line[1:]

features = 784


def expnd(line, num):
  full = [ '0' for i in range(num) ]
  for val in line:
    val = val.split(":")
    full[int(val[0])] = val[1]
  return full

fd=open("../svmLightTrainingData.txt")
fo=open("train_f_op.txt","w")
fi=open("train_f_ip.txt","w")
for lines in fd.readlines():
  lines = lines.strip().split()
  if ec[lines[0]] == []:
    print "here"
  fo.write(" ".join(ec[lines[0]]) + "\n")
  fi.write(" ".join(expnd(lines[1:], features)) + "\n")
  #fi.write(" ".join(lines[1:]) + "\n")
fo.close()
fi.close()
fd.close()
 
fd=open("../svmLightTestingData.txt")
fo=open("test_f_op.txt","w")
fi=open("test_f_ip.txt","w")
for lines in fd.readlines():
  lines = lines.strip().split()
  if ec[lines[0]] == []:
    print "here"
  #fo.write(" ".join(ec[lines[0]]) + "\n")
  fo.write(lines[0] + "\n")
  #fi.write(" ".join(lines[1:]) + "\n")
  fi.write(" ".join(expnd(lines[1:], features)) + "\n")
fi.close()
fd.close()
fo.close()
