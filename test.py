import environment
import numpy as np
import tensorflow as tf

h_fc5 = np.array([[0,1,2,3,4,20,6,7,8,9,10,11,12,13,14,15,16,17,18,19],
                  [20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39],
                  [40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59]])
h_fc5_reshape = np.reshape(h_fc5,newshape=[-1,4,5])
print(h_fc5_reshape)

aaa = np.array([[0,4,3,2],[0,4,3,2],[0,4,3,2]])

res = [h_fc5_reshape[:,:,i] for i in aaa]

print(res)


