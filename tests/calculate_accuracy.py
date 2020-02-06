""" 
calculate_accuracy_singleTraining.py: calculates accuracy of a model trained at a specific loss
and phase uncertainty - now w/ separate phase uncert for theta and phi

Author: Simon Geoffroy-Gagnon
Edit: 29.01.2020
"""
import numpy as np
import sys
from time import sleep
from tqdm import tqdm
import scipy.io
sys.path.append('/home/simon/Documents/neuroptica')
import neuroptica as neu

def get_accuracy(ONN, model, Xt, yt):
    pbar = tqdm(total=len(ONN.loss_dB)*len(ONN.phase_uncert_theta))
    accuracy = []
    for loss_dB in ONN.loss_dB:
        pbar.set_description(f'Loss/MZI (dB): {loss_dB:.2f}/{ONN.loss_dB[-1]:.2f}', refresh=True)
        acc_theta = []
        for phase_uncert_theta in ONN.phase_uncert_theta:
            acc_phi = []
            for phase_uncert_phi in ONN.phase_uncert_phi:
                acc = []    
                for _ in range(ONN.ITERATIONS):
                    model.set_all_phases_uncerts_losses(ONN.Phases[-1], phase_uncert_theta, phase_uncert_phi, loss_dB, ONN.loss_diff)
                    Y_hat = model.forward_pass(Xt.T)
                    pred = np.array([np.argmax(yhat) for yhat in Y_hat.T])
                    gt = np.array([np.argmax(tru) for tru in yt])
                    acc.append(np.sum(pred == gt)/yt.shape[0]*100)
                acc_phi.append(np.mean(acc)) 
            acc_theta.append(acc_phi)
            pbar.update(1)
        accuracy.append(acc_theta)
    pbar.close()
    return np.swapaxes(np.array(accuracy), 0, 2)

if __name__ == '__main__':
    iterator = tqdm(total=10)
    for epoch in range(10):
        iterator.set_description(f"ℒ = {epoch}", refresh=True)
        iterator.update(1)
        sleep(.1)
        


