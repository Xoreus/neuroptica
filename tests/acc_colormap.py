import matplotlib.pyplot as plt
import numpy as np
import matplotlib

def Loss_PhaseUncert(onn):
    plt.pcolor(onn.loss_dB, onn.phase_uncert_phi, np.squeeze(onn.accuracy_LPU), cmap='inferno', clim=[100/(onn.N+1),100])
    plt.title(onn.topology)
    plt.xlabel('Loss/MZI (dB)')
    plt.ylabel(r'$\sigma_\phi = \sigma_\theta$ (Rad)')
    plt.savefig(f'{onn.FOLDER}/LPU_{onn.topology}_N={onn.N}.png')
    plt.clf()


def PhiTheta(onn):
    plt.pcolor(onn.phase_uncert_phi, onn.phase_uncert_theta, np.squeeze(onn.accuracy_PT), cmap='inferno', clim=[100/(onn.N+1),100])
    plt.title(onn.topology)
    plt.ylabel(r'$\sigma_{\phi}$ (Rad)')
    plt.xlabel(r'$\sigma_\theta$ (Rad)')
    plt.savefig(f'{onn.FOLDER}/PT_{onn.topology}_N={onn.N}.png')
    plt.clf()


def cube_plotting(onn, loss_idx=0, pu=0):
    phiTheta = onn.accuracy[:,:,loss_idx]
    plt.pcolor(onn.phase_uncert_phi, onn.phase_uncert_theta, phiTheta, cmap='magma', clim=[100/(onn.N+1),100])
    plt.title(onn.topology)
    plt.ylabel(r'$\sigma_{\phi}$ (Rad)')
    plt.xlabel(r'$\sigma_\theta$ (Rad)')
    plt.savefig(f'{onn.FOLDER}/PT_{onn.topology}_N={onn.N}.png')
    plt.clf()
    
    samePU = [] 
    for loss_idx in range(onn.accuracy.shape[0]):
        samePU.append(onn.accuracy[loss_idx, loss_idx, :])
    plt.pcolor(onn.loss_dB, onn.phase_uncert_phi, samePU, cmap='viridis', clim=[100/(onn.N+1),100]) 
    plt.title(onn.topology)
    plt.xlabel('Loss/MZI (dB)')
    plt.ylabel(r'$\sigma_\phi = \sigma_\theta$ (Rad)')
    plt.savefig(f'{onn.FOLDER}/LPU_{onn.topology}_N={onn.N}.png')
    plt.clf()

def colormap_me(onn):
    PhiTheta(onn)
    Loss_PhaseUncert(onn)


