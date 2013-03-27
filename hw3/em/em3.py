
# try these inputs:
#  python2.6 three_coins.py 5 0.3 0.3 0.6 HHH TTT HHH TTT HHH
#  python2.6 three_coins.py 5 0.3 0.3 0.6 HHH TTT HHH TTT
#  python2.6 three_coins.py 5 0.3 0.7 0.7 HHH TTT HHH TTT 
#  python2.6 three_coins.py 11 0.3 0.7001 0.7 HHH TTT HHH TTT 
#  python2.6 three_coins.py 11 0.3 0.6999 0.7 HHH TTT HHH TTT 

import math
import sys

# number of iterations for EM
num_iters = 1000

# coin 0 has probability pi
pi = 0.8

# coin 1 has probability p of heads
#p = 0.6999
p = 0.75

# coin 2 has probability r of heads
#r = 0.7
r = 0.4

# First toss coin 0
# if coin 0 == H then toss coin 1 three times
# if coin 0 == T then toss coin 2 three times

#observations = ["HHH", "TTT", "HHH", "TTT"]
observations = ["HHH", "TTT", "HHH", "TTT", "HHH"]

if len(sys.argv) > 5:
  num_iters = int(sys.argv[1])
  pi = float(sys.argv[2])
  p = float(sys.argv[3])
  r = float(sys.argv[4])
  observations = sys.argv[5:]

print "num_iters =", num_iters
print "pi =", pi
print "p =", p
print "r =", r
#print "observations =", observations
print

def observed(x, y):
  if len(x) < 1: 
    return 0.0
  prob = 1.0
  for i in x:
    if i == 'H':
      if y == 'H': prob = prob * p
      else: prob = prob * r
    if i == 'T':
      if y == 'H': prob = prob * (1 - p)
      else: prob = prob * (1 - r)
  return prob

def posterior(y, x):
  return (pi * observed(x, y)) / (pi*observed(x, 'H') + (1-pi)*observed(x, 'T'))

def expected_counts(observations, print_posterior=0):
  count = 0.0
  count_p = 0.0
  total_p = 0.0
  count_r = 0.0
  total_r = 0.0
  total = len(observations)
  for obs in observations:
    for y in "HT": 
      post = posterior(y, obs)
      if print_posterior == 1: print y, obs, post
      if y == 'H': 
        count += post
      for x in obs:
        if y == 'H' and x == 'H': 
          count_p += post
          total_p += post
        if y == 'H' and x == 'T': 
          total_p += post
        if y == 'T' and x == 'H': 
          count_r += post
          total_r += post
        if y == 'T' and x == 'T': 
          total_r += post
  print
  print "Iteration", i
  print "pi =", count / total
  print "p =", count_p / total_p
  print "r =", count_r / total_r
  return (count / total, count_p / total_p, count_r / total_r)

for i in range(1,num_iters+1):
  (pi, p, r) = expected_counts(observations, 0)

