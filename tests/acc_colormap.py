import matplotlib.pyplot as plt
import numpy as np
import matplotlib
from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
matplotlib.rcParams['mathtext.fontset'] = 'custom'
matplotlib.rcParams['mathtext.rm'] = 'Bitstream Vera Sans'
matplotlib.rcParams['mathtext.it'] = 'Bitstream Vera Sans:italic'
matplotlib.rcParams['mathtext.bf'] = 'Bitstream Vera Sans:bold'

def PU(onn):
    plt.pcolor(onn.loss_dB, onn.phase_uncert_phi, np.squeeze(onn.accuracy), cmap='viridis')
    plt.title(onn.topology)
    plt.xlabel('Loss/MZI (dB)')
    plt.ylabel(r'$\sigma_\phi = \sigma_\theta$ (Rad)')
    plt.savefig(f'{onn.FOLDER}/Loss+PU_{onn.topology}_N={onn.N}.png')


def PhiTheta(onn):
    plt.pcolor(onn.phase_uncert_phi, onn.phase_uncert_theta, np.squeeze(onn.accuracy), cmap='magma')
    plt.title(onn.topology)
    plt.ylabel(r'$\sigma_{\phi}$ (Rad)')
    plt.xlabel(r'$\sigma_\theta$ (Rad)')
    plt.savefig(f'{onn.FOLDER}/PhiTheta_{onn.topology}_N={onn.N}.png')

def cube_plotting(onn, loss_idx=0, pu=0):

    phiTheta = onn.accuracy[:,:,loss_idx]
    plt.pcolor(onn.phase_uncert_phi, onn.phase_uncert_theta, phiTheta, cmap='magma')
    plt.title(onn.topology)
    plt.ylabel(r'$\sigma_{\phi}$ (Rad)')
    plt.xlabel(r'$\sigma_\theta$ (Rad)')
    plt.savefig(f'{onn.FOLDER}/PhiTheta_{onn.topology}_N={onn.N}.png')

    samePU = [] 
    for loss_idx in range(onn.accuracy.shape[0]):
        samePU.append(onn.accuracy[loss_idx, loss_idx, :])
    plt.pcolor(onn.loss_dB, onn.phase_uncert_phi, samePU, cmap='viridis')
    plt.title(onn.topology)
    plt.xlabel('Loss/MZI (dB)')
    plt.ylabel(r'$\sigma_\phi = \sigma_\theta$ (Rad)')
    plt.savefig(f'{onn.FOLDER}/Loss+PU_{onn.topology}_N={onn.N}.png')
