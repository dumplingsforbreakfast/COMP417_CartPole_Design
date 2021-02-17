# Starter code for those trying to use LQR. Your
# K matrix controller should come from a call to lqr(A,B,Q,R),
# which we have provided. Below this are "dummy" matrices of the right
# type and size. If you fill in these with values you derive by hand
# they should work correctly to call the function.

# Here is the provided LQR function
from typing import Optional, Any
import scipy
import numpy as np
import math

def lqr(A, B, Q, R):
    x = scipy.linalg.solve_continuous_are(A, B, Q, R)
    k = np.linalg.inv(R) * np.dot(B.T, x)
    return k



# FOR YOU TODO: Fill in the values for A, B, Q and R here.
# Note that they should be matrices not scalars. 
# Then, figure out how to apply the resulting k
# to solve for a control, u, within the policyfn that balances the cartpole.
A = np.array([[0, 1, 0, 0],
              [0, -1.6, 0 , 5.892], #[0, -4b/(4M+m), 0, 3mg/(4M+m)]
              [0, -4.8, 0, 47.136],#[0, -6b/(4lM+lm), 0, 6(m+M)g/(4LM+lm)]
              [0, 0, 1, 0]])
##B = np.array( [[0, 4/(4*M+m), 6/(4*l*M+l*m), 0 ]] )
B = np.array([[0, 1.6, 4.8, 0]])
B.shape = (4, 1)

Q = np.array([[20, 0, 0, 0],
              [0, 5, 0, 0],
              [0, 0, 10, 0],
              [0, 0, 0, 100]])

R = np.array([[0.1]])

print("A holds:", A)
print("B holds:", B)
print("Q holds:", Q)
print("R holds:", R)

# Uncomment this to get the LQR gains k once you have
# filled in the correct matrices.
#k = lqr(A, B, Q, R)
#print("k holds:", k)


