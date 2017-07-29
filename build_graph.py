import matplotlib.pyplot as plt
import numpy as np

lines = open('./output.txt','r').readlines()

errs = []
for line in lines:
	err = line.split('err:')
	if len(err)>1:
		errs.append(float(err[1]))

errs = np.array(errs)

plt.plot(errs)
plt.gca().set_ylim(bottom=0)
plt.gca().set_ylabel("Test Error (%)")
plt.gca().set_xlabel("Epoch")
plt.savefig("error.png")
plt.close()