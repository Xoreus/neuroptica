import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

fg_rv = np.sqrt(2)/np.sqrt(np.pi)
loss_mean = 0.5

sz = 500000 # Number of samples taken
b = 100 # Bins

for scale in [0, 0.25]:
    for loss_min in [0,  0.25]:
        shape = 2
        f = np.random.gamma(shape, scale, sz);

        actual_loss_f = loss_min + f;

        plt.figure(figsize=(10,10))
        plt.hist(actual_loss_f, bins=b, density=True)
        plt.xlabel('Loss/MZI (dB)', fontdict = {'fontsize' : 20})
        plt.ylabel('Frequency (normalized)', fontdict = {'fontsize' : 20})
        plt.xlim([-0.5, 3])
        ax = plt.gca()
        ax.tick_params(axis = 'both', which = 'major', labelsize = 20)
        ax.tick_params(axis = 'both', which = 'minor', labelsize = 20)
        plt.tight_layout()
        plt.savefig(f'Figures/gamma-shape={shape}-scale={scale}-min={min(actual_loss_f):.3f}_mean={np.mean(actual_loss_f):.3f}.pdf')


