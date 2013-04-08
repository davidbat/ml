import cvxopt, svmcmpl
m = 2000
X = 2.0*cvxopt.uniform(m,2)-1.0
d = cvxopt.matrix([2*int(v>=0)-1 for v in cvxopt.mul(X[:,0],X[:,1])],(m,1))
gamma = 2.0; kernel = 'rbf'; sigma = 1.0; width = 20
sol1 = svmcmpl.softmargin(X, d, gamma, kernel, sigma)
sol2 = svmcmpl.softmargin_appr(X, d, gamma, width, kernel, sigma)