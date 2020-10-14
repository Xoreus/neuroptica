import numpy as np
import matplotlib.pyplot as plt

loss_diff = 0.25;
loss_dB = 0.25;

shape_g = loss_diff*8; # Gamma process standard dev = ??
scale_g = loss_diff * 1
mu_e = loss_diff; # Exponential 'Standard dev'
sigma_g = loss_diff; # Gaussian standard dev
sigma_f = loss_diff/(np.sqrt(2)/np.sqrt(np.pi)) # Folded Gaussian standard dev

sz = 500000 # Number of samples taken
b = 100 # Bins

e = np.random.exponential(mu_e, sz);
f = abs(np.random.normal(0, sigma_f, sz));
g = (np.random.normal(loss_dB*2, sigma_g, sz));
g = np.random.gamma(shape_g, scale_g, sz) 

actual_loss_e = loss_dB + e;
actual_loss_f = loss_dB + f;
actual_loss_g = abs(g);
actual_loss_g = g

plt.hist(actual_loss_e, bins=b, density=True)
plt.title(f'Hist, Exponential RV, Loss_dB Min = {np.min(actual_loss_e):.3f}\nLoss_dB Mean = {np.mean(actual_loss_e):.3f} dB')
plt.savefig('exponential.png')
plt.xlabel('Loss/MZI (dB)')
plt.clf()

plt.hist(actual_loss_g, bins=b, density=True)
plt.title(f'Hist, Gaussian RV, Loss_dB Min = {np.min(actual_loss_g):.3f}\nLoss_dB Mean = {np.mean(actual_loss_g):.3f}')
plt.xlabel('Loss/MZI (dB)')
plt.savefig('gaussian.png')
plt.clf()

plt.hist(actual_loss_f, bins=b, density=True)
plt.title(f'Hist, Folded Gaussian RV, Loss_dB Min = {np.min(actual_loss_f):.3f}\nLoss_dB Mean = {np.mean(actual_loss_f):.3f}')
plt.xlabel('Loss/MZI (dB)')
plt.savefig('folded.png')
plt.clf()

plt.hist(actual_loss_g, bins=b, density=True)
plt.title(f'Hist, Gamma RV, Loss_dB Min = {np.min(actual_loss_g):.3f}\nLoss_dB Mean = {np.mean(actual_loss_g):.3f}')
plt.xlabel('Loss/MZI (dB)')
plt.savefig('gamma.png')
plt.clf()

 
