"""
calculate_accuracy_singleTraining.py: calculates accuracy of a model trained at a specific loss
and phase uncertainty - now w/ separate phase uncert for theta and phi

Author: Simon Geoffroy-Gagnon
Edit: 2020.03.04
"""
import numpy as np
from tqdm import tqdm
import time
# from main import truncate_diamond_sigma_adjust

def get_accuracy_PT(ONN, model, Xt, yt, loss_diff=0, show_progress=True):
    ONN.PT_Area = (ONN.phase_uncert_phi[1] - ONN.phase_uncert_phi[0])**2
    t = time.time()

    if show_progress: pbar = tqdm(total=len(ONN.phase_uncert_phi)*len(ONN.phase_uncert_theta))
    accuracy = []
    for loss_dB in ONN.loss_dB[:1]:
        # print(loss_dB)
        acc_theta = []
        for phase_uncert_theta in ONN.phase_uncert_theta:
            if show_progress: pbar.set_description(f'Theta phase uncert = {phase_uncert_theta:.2f}/{ONN.phase_uncert_theta[-1]:.2f}', refresh=True)
            acc_phi = []
            for phase_uncert_phi in ONN.phase_uncert_phi:
                acc = []
                for _ in range(ONN.ITERATIONS):
                    model.set_all_phases_uncerts_losses(ONN.phases, phase_uncert_theta, phase_uncert_phi, loss_dB, loss_diff)
                    # sigma_adjust(model) # should adjustments needed
                    Y_hat = model.forward_pass(Xt.T)
                    pred = np.array([np.argmax(yhat) for yhat in Y_hat.T])
                    gt = np.array([np.argmax(tru) for tru in yt])
                    acc.append(np.sum((pred == gt))/yt.shape[0]*100)
                acc_phi.append(np.mean(acc))
                if show_progress: pbar.update(1)
            acc_theta.append(acc_phi)
        accuracy.append(acc_theta)
    if show_progress: pbar.close()
    return np.squeeze(np.swapaxes(np.array(accuracy), 0, 2))

def get_accuracy_LPU(ONN, model, Xt, yt, loss_diff=0, show_progress=True):
    ONN.LPU_Area = (ONN.loss_dB[1] - ONN.loss_dB[0])*(ONN.phase_uncert_phi[1] - ONN.phase_uncert_phi[0])
    t = time.time()

    if show_progress: pbar = tqdm(total=len(ONN.loss_dB)*len(ONN.phase_uncert_theta))
    accuracy = []
    for loss_dB in ONN.loss_dB:
        if show_progress: pbar.set_description(f'Loss/MZI (dB): {loss_dB:.2f}/{ONN.loss_dB[-1]:.2f}', refresh=True)
        acc_theta = []
        for phase_uncert_theta in ONN.phase_uncert_theta:
            acc_phi = []
            ONN.phase_uncert_phi_curr = [phase_uncert_theta]
            for phase_uncert_phi in ONN.phase_uncert_phi_curr:
                acc = []
                for _ in range(ONN.ITERATIONS):
                    model.set_all_phases_uncerts_losses(ONN.phases, phase_uncert_theta, phase_uncert_phi, loss_dB, loss_diff)
                    # sigma_adjust(model) # should adjustments needed
                    Y_hat = model.forward_pass(Xt.T)
                    pred = np.array([np.argmax(yhat) for yhat in Y_hat.T])
                    gt = np.array([np.argmax(tru) for tru in yt])
                    acc.append(np.sum((pred == gt))/yt.shape[0]*100)
                # if (phase_uncert_theta==0 and phase_uncert_phi==0 and loss_dB==0.1):
                #     print("MAX FOR ITERATION")
                #     print(max(acc))
                acc_phi.append(np.mean(acc))
            acc_theta.append(acc_phi)
            if show_progress: pbar.update(1)
        accuracy.append(acc_theta)
    if show_progress: pbar.close()
    return np.squeeze(np.swapaxes(np.array(accuracy), 0, 2))

def get_accuracy_SLPU(ONN, model, Xt, yt, loss_diff=0, show_progress=True):
    t = time.time()

    if show_progress: pbar = tqdm(total=len(ONN.loss_dB))
    accuracy = []
    for loss_dB in ONN.loss_dB:
        if show_progress: pbar.set_description(f'Loss/MZI (dB): {loss_dB:.2f}/{ONN.loss_dB[-1]:.2f}', refresh=True)
        acc = []
        for _ in range(ONN.ITERATIONS):
            model.set_all_phases_uncerts_losses(ONN.phases, 0, 0, loss_dB, loss_diff)
            Y_hat = model.forward_pass(Xt.T)
            pred = np.array([np.argmax(yhat) for yhat in Y_hat.T])
            gt = np.array([np.argmax(tru) for tru in yt])
            acc.append(np.sum((pred == gt))/yt.shape[0]*100)
        if show_progress: pbar.update(1)
        accuracy.append(np.mean(acc))
    if show_progress: pbar.close()
    return np.array(accuracy)
    
def get_accuracy(onn, model, Xt, yt, loss_diff, show_progress=True):
    if onn.same_phase_uncert:
       return get_accuracy_LPU(onn, model, Xt, yt, loss_diff=loss_diff, show_progress=show_progress)
    else:
       return get_accuracy_PT(onn, model, Xt, yt, loss_diff=loss_diff, show_progress=show_progress)

def accuracy(onn, model, Xt, yt):
    model.set_all_phases_uncerts_losses(onn.phases)
    Y_hat = model.forward_pass(Xt.T)
    pred = np.array([np.argmax(yhat) for yhat in Y_hat.T])
    gt = np.array([np.argmax(tru) for tru in yt])
    acc = np.sum((pred == gt))/yt.shape[0]*100
    return acc
