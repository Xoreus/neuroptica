import numpy as np
import matplotlib.pyplot as plt

fg_rv = np.sqrt(2)/np.sqrt(np.pi)
loss_mean = 0.5

sz = 500000 # Number of samples taken
b = 100 # Bins

loss_diff = 0
loss_min = loss_mean - loss_diff
sigma_f = loss_diff/fg_rv # Folded Gaussian standard dev
f = abs(np.random.normal(0, sigma_f, sz));
actual_loss_f = loss_min + f;

plt.figure(figsize=(6,6))
plt.hist(actual_loss_f, bins=b, density=True)
plt.title(f'Hist, Folded Gaussian RV, Loss_dB Min = {np.min(actual_loss_f):.3f}\nLoss_dB Mean = {np.mean(actual_loss_f):.3f}', fontdict = {'fontsize' : 20})
plt.xlabel('Loss/MZI (dB)', fontdict = {'fontsize' : 20})
plt.ylabel('Frequency (normalized)', fontdict = {'fontsize' : 20})
plt.xlim([0, 2])
# plt.yticks([])
ax = plt.gca()
ax.tick_params(axis = 'both', which = 'major', labelsize = 15)
ax.tick_params(axis = 'both', which = 'minor', labelsize = 15)
plt.savefig(f'folded_min={np.min(actual_loss_f):.3f}.pdf')
plt.clf()


loss_diff = 0.4
loss_min = loss_mean - loss_diff
sigma_f = loss_diff/fg_rv # Folded Gaussian standard dev
f = abs(np.random.normal(0, sigma_f, sz));
actual_loss_f = loss_min + f;

sigma_f = loss_diff/(np.sqrt(2)/np.sqrt(np.pi)) # Folded Gaussian standard dev
plt.figure(figsize=(6,6))
plt.hist(actual_loss_f, bins=b, density=True)
# plt.title(f'Hist, Folded Gaussian RV, Loss_dB Min = {np.min(actual_loss_f):.3f}\nLoss_dB Mean = {np.mean(actual_loss_f):.3f}')
plt.xlabel('Loss/MZI (dB)', fontdict = {'fontsize' : 20})
plt.ylabel('Frequency (normalized)', fontdict = {'fontsize' : 20})
plt.xlim([0, 2])
ax = plt.gca()
ax.tick_params(axis = 'both', which = 'major', labelsize = 15)
ax.tick_params(axis = 'both', which = 'minor', labelsize = 15)
plt.savefig(f'folded_min={np.min(actual_loss_f):.3f}.pdf')
plt.clf()


loss_diff = 0.025

loss_min = loss_mean - loss_diff
sigma_f = loss_diff/fg_rv # Folded Gaussian standard dev
f = abs(np.random.normal(0, sigma_f, sz));
actual_loss_f = loss_min + f;

sigma_f = loss_diff/(np.sqrt(2)/np.sqrt(np.pi)) # Folded Gaussian standard dev
plt.figure(figsize=(6,6))
plt.hist(actual_loss_f, bins=b, density=True)
# plt.title(f'Hist, Folded Gaussian RV, Loss_dB Min = {np.min(actual_loss_f):.3f}\nLoss_dB Mean = {np.mean(actual_loss_f):.3f}')
plt.xlabel('Loss/MZI (dB)', fontdict = {'fontsize' : 20})
plt.ylabel('Frequency (normalized)', fontdict = {'fontsize' : 20})
plt.xlim([0, 2])
ax = plt.gca()
ax.tick_params(axis = 'both', which = 'major', labelsize = 15)
ax.tick_params(axis = 'both', which = 'minor', labelsize = 15)
plt.savefig(f'folded_min={np.min(actual_loss_f):.3f}.pdf')
plt.clf()


