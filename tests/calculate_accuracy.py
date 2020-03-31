"""
calculate_accuracy_singleTraining.py: calculates accuracy of a model trained at a specific loss
and phase uncertainty - now w/ separate phase uncert for theta and phi

Author: Simon Geoffroy-Gagnon
Edit: 2020.03.04
"""
import numpy as np
from tqdm import tqdm
import time

def get_accuracy_singleLoss(ONN, model, Xt, yt, loss_diff=0):
    ONN.PT_Area = (ONN.phase_uncert_phi[1] - ONN.phase_uncert_phi[0])**2
    t = time.time()
    if 'C' in ONN.topo and 'Q' in ONN.topo:
        Xt = np.array([list(np.zeros(int((ONN.N-2)))) + list(samples) for samples in ONN.Xt])
    elif 'C' in ONN.topo and 'W' in ONN.topo:
        Xt = (np.array([list(np.zeros(int((ONN.N-2)/2))) + list(samples) +
            list(np.zeros(int(np.ceil((ONN.N-2)/2)))) for samples in ONN.Xt]))

    pbar = tqdm(total=len(ONN.phase_uncert_phi)*len(ONN.phase_uncert_theta))
    accuracy = []
    for loss_dB in ONN.loss_dB[:1]:
        acc_theta = []
        for phase_uncert_theta in ONN.phase_uncert_theta:
            pbar.set_description(fr'theta PU = {phase_uncert_theta:.2f}/{ONN.phase_uncert_theta[-1]:.2f}', refresh=True)
            acc_phi = []
            for phase_uncert_phi in ONN.phase_uncert_phi:
                acc = []
                for _ in range(ONN.ITERATIONS):
                    model.set_all_phases_uncerts_losses(ONN.phases, phase_uncert_theta, phase_uncert_phi, loss_dB, loss_diff)
                    Y_hat = model.forward_pass(Xt.T)
                    pred = np.array([np.argmax(yhat) for yhat in Y_hat.T])
                    gt = np.array([np.argmax(tru) for tru in yt])
                    acc.append(np.sum((pred == gt))/yt.shape[0]*100)
                acc_phi.append(np.mean(acc))
                pbar.update(1)
            acc_theta.append(acc_phi)
        accuracy.append(acc_theta)
    pbar.close()
    return np.squeeze(np.swapaxes(np.array(accuracy), 0, 2))

def get_accuracy_samePU(ONN, model, Xt, yt, loss_diff=0):
    ONN.LPU_Area = (ONN.loss_dB[1] - ONN.loss_dB[0])*(ONN.phase_uncert_phi[1] - ONN.phase_uncert_phi[0])
    t = time.time()
    if 'C' in ONN.topo and 'Q' in ONN.topo:
        Xt = np.array([list(np.zeros(int((ONN.N-2)))) + list(samples) for samples in ONN.Xt])
    elif 'C' in ONN.topo and 'W' in ONN.topo:
        Xt = (np.array([list(np.zeros(int((ONN.N-2)/2))) + list(samples) +
            list(np.zeros(int(np.ceil((ONN.N-2)/2)))) for samples in ONN.Xt]))

    pbar = tqdm(total=len(ONN.loss_dB)*len(ONN.phase_uncert_theta))
    accuracy = []
    for loss_dB in ONN.loss_dB:
        pbar.set_description(f'Loss/MZI (dB): {loss_dB:.2f}/{ONN.loss_dB[-1]:.2f}', refresh=True)
        acc_theta = []
        for phase_uncert_theta in ONN.phase_uncert_theta:
            acc_phi = []
            ONN.phase_uncert_phi_curr = [phase_uncert_theta]
            for phase_uncert_phi in ONN.phase_uncert_phi_curr:
                acc = []
                for _ in range(ONN.ITERATIONS):
                    model.set_all_phases_uncerts_losses(ONN.phases, phase_uncert_theta, phase_uncert_phi, loss_dB, loss_diff)
                    Y_hat = model.forward_pass(Xt.T)
                    pred = np.array([np.argmax(yhat) for yhat in Y_hat.T])
                    gt = np.array([np.argmax(tru) for tru in yt])
                    acc.append(np.sum((pred == gt))/yt.shape[0]*100)
                acc_phi.append(np.mean(acc))
            acc_theta.append(acc_phi)
            pbar.update(1)
        accuracy.append(acc_theta)
    pbar.close()
    return np.squeeze(np.swapaxes(np.array(accuracy), 0, 2))

def get_accuracy(onn, model, Xt, yt, loss_diff):
    if onn.same_phase_uncert:
       return get_accuracy_samePU(onn, model, Xt, yt, loss_diff=loss_diff)
    else:
       return get_accuracy_singleLoss(onn, model, Xt, yt, loss_diff=loss_diff)
        
