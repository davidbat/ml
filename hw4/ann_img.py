

def ReadFile_hash(fn, jumps):
	print "Reading file - ", fn
	fd = open(fn)
	lines = fd.readlines()
	hashes = []
	i = -1
	for line in lines:
		i += 1
		if i % jumps != 0:
			continue
		
		#print i
		tmp_hash = {}
		line = line.split()
		tmp_hash['y'] = float(line[0])
		for item in line [1:]:
			item = item.split(":")
			# keys are ints and values are floats
			tmp_hash[int(item[0])] = float(item[1])
		hashes.append(tmp_hash)
	print "Finished reading the data"
	return hashes