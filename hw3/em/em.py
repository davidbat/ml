import sys
import math

fn = sys.argv[1]
pi = float(sys.argv[2])
Wa = float(sys.argv[3])
Wb = float(sys.argv[4])

fd = open(fn)
flips = fd.readlines()
fd.close()
converged = False


def nCr(n,r):
    f = math.factorial
    return f(n) / f(r) / f(n-r)

#if __name__ == '__main__':
print nCr(4,2)
t=0

while not converged and t < 100:
	t += 1
	H1 = 0
	T1 = 0
	H2 = 0
	T2 = 0
	for line in flips:
		H = line.count("H")
		T = line.count("T")
		seq_len = H+T
		prob1 = 0.8 * (Wa ** H) * ((1 - Wa) ** T)
		prob2 = 0.2 * (Wb ** H) * ((1 - Wb) ** T)
		prb1 = prob1 / (prob1 + prob2)
		prb2 = prob2 / (prob1 + prob2)
#		print H, T, prob1, prob2
		H1 += prb1 * H
		T1 += prb1 * T
		H2 += prb2 * H
		T2 += prb2 * T
	print H1, H2, T1 , T2
	Wa = H1 / (H1 + T1)
	Wb = H2 / (H2 + T2)

