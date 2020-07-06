import numpy as np
import matplotlib.pyplot as plt

loss_mean = 0
std_dev = 0.25

sz = 500000 # Number of samples taken
b = 100 # Bins

f = np.random.normal(0, std_dev, sz)
actual_loss_f = f

plt.figure(figsize=(7,7))
plt.hist(actual_loss_f, bins=b, density=True)
plt.xlabel('Phase Error (Rad)', fontdict = {'fontsize' : 20})
plt.ylabel('Frequency (normalized)', fontdict = {'fontsize' : 20})
plt.xlim([-2, 2])
# plt.yticks([])
ax = plt.gca()
ax.tick_params(axis = 'both', which = 'major', labelsize = 15)
ax.tick_params(axis = 'both', which = 'minor', labelsize = 15)
plt.savefig(f'Figures/gauss_min={np.min(actual_loss_f):.3f}.pdf')
plt.clf()

