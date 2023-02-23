import pandas as pd
import numpy as np
from keras.models import Sequential, load_model

c = np.load("test1.npy")
print(c)
a0 = c[0]
a1 = c[1]
a2 = c[2]
b = np.zeros((5, 3, 7))
for i in range(5):
    b[i][0] = a0[i]
    b[i][1] = a1[i]
    b[i][2] = a2[i]
model = load_model("F:\python\workspace\B-GAP-H/test\lstm/traj_model_120.h5")
print(b.shape)
pre = model.predict(b)
print(pre)