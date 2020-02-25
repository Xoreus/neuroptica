"""
calculate_accuracy_singleTraining.py: calculates accuracy of a model trained at a specific loss
and phase uncertainty - now w/ separate phase uncert for theta and phi

Author: Simon Geoffroy-Gagnon
Edit: 29.01.2020
"""
import numpy as np
from tqdm import tqdm
import time


def get_accuracy(ONN, model, Xt, yt, loss_diff=0):
    t = time.time()
    if 'C' in ONN.onn_topo and 'Q' in ONN.onn_topo:
        Xt = np.array([list(np.zeros(int((ONN.N-2)))) + list(samples) for samples in ONN.Xt])
    elif 'C' in ONN.onn_topo and 'W' in ONN.onn_topo:
        Xt = (np.array([list(np.zeros(int((ONN.N-2)/2))) + list(samples) +
            list(np.zeros(int(np.ceil((ONN.N-2)/2)))) for samples in ONN.Xt]))

    pbar = tqdm(total=len(ONN.loss_dB)*len(ONN.phase_uncert_theta))
    accuracy = []
    for loss_dB in ONN.loss_dB:
        pbar.set_description(f'Loss/MZI (dB): {loss_dB:.2f}/{ONN.loss_dB[-1]:.2f}', refresh=True)
        acc_theta = []
        for phase_uncert_theta in ONN.phase_uncert_theta:
            acc_phi = []
            if ONN.same_phase_uncert:
                ONN.phase_uncert_phi = [phase_uncert_theta]
            for phase_uncert_phi in ONN.phase_uncert_phi:
                acc = []
                for _ in range(ONN.ITERATIONS):
                    model.set_all_phases_uncerts_losses(ONN.phases, phase_uncert_theta, phase_uncert_phi, loss_dB, loss_diff)
                    Y_hat = model.forward_pass(Xt.T)
                    pred = np.array([np.argmax(yhat) for yhat in Y_hat.T])
                    # zeta_filter = np.array([sorted(yhat)[-1] - sorted(yhat)[-2] > ONN.zeta for yhat in Y_hat.T])

                    gt = np.array([np.argmax(tru) for tru in yt])
                    # acc.append(np.sum((pred == gt)*zeta_filter)/yt.shape[0]*100)
                    acc.append(np.sum((pred == gt))/yt.shape[0]*100)
                acc_phi.append(np.mean(acc))
            acc_theta.append(acc_phi)
            pbar.update(1)
        accuracy.append(acc_theta)
    pbar.close()
    return np.swapaxes(np.array(accuracy), 0, 2)
