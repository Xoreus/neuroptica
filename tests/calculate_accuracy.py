""" 
calculate_accuracy_singleTraining.py: calculates accuracy of a model trained at a specific loss
and phase uncertainty

Author: Simon Geoffroy-Gagnon
Edit: 29.01.2020
"""
import numpy as np
import sys
from time import sleep
from tqdm import tqdm
sys.path.append('/home/simon/Documents/neuroptica')
import neuroptica as neu

def get_accuracy(ONN, model):
    pbar = tqdm(total=len(ONN.loss_dB))
    accuracy = []
    for loss_dB in ONN.loss_dB:
        pbar.set_description(f'Loss: {loss_dB:.2f}/{ONN.loss_dB[-1]:.2f}', refresh=True)
        acc_array = []
        for phase_uncert in ONN.phase_uncert:
            acc = []    
            for _ in range(ONN.ITERATIONS):
                model.set_all_phases_uncerts_losses(ONN.Phases[-1], phase_uncert, loss_dB, ONN.loss_diff)
                Y_hat = model.forward_pass(ONN.Xt.T)
                pred = np.array([np.argmax(yhat) for yhat in Y_hat.T])
                gt = np.array([np.argmax(tru) for tru in ONN.yt])
                acc.append(np.sum(pred == gt)/ONN.yt.shape[0]*100)
            acc_array.append(np.mean(acc))
        accuracy.append(acc_array)
        pbar.update(1)

    pbar.close()
    return accuracy

if __name__ == '__main__':
    iterator = tqdm(total=10)
    for epoch in range(10):
        iterator.set_description(f"ℒ = {epoch}", refresh=True)
        iterator.update(1)
        sleep(.1)
        


