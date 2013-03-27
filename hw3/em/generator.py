import sys
import random


if len(sys.argv) == 5:
	n, pi, p, r = sys.argv[1:]
	n = int(n)
	pi = float(pi)
	p = float(p)
	r = float(r)
else:
	n = 1000
	pi = 0.8
	p = 0.75
	r = 0.4

seq_len = 10
digits = 100
pi_arr = ([ 1 ] * int(pi * digits)) + ([ 2 ] * int(digits * (1.0 - pi)))
random.shuffle(pi_arr)
p_arr = ([ 'H' ] * int(p * digits)) + ([ 'T' ] * int(digits * (1.0 - p)))
random.shuffle(p_arr)
r_arr = ([ 'H' ] * int(r * digits)) + ([ 'T' ] * int(digits * (1.0 - r)))
random.shuffle(r_arr)
#print pi_arr

for i in range(n):
	coin = random.choice(pi_arr)
	if coin == 1:
		prob = p_arr
	else:
		prob = r_arr
	dist = ""
	for i in range(seq_len):
		dist += random.choice(prob)
	print dist

