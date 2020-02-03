""" 
Function to save all of the simulation data at every loss/phase_uncert training setting

Author: Simon Geoffroy-Gagnon
Edit: 17.01.2020
"""
import sys
import numpy as np
import pandas as pd
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
import plot_scatter_matrix as psm
import matplotlib as mpl
mpl.use('Agg')

def saveSimData(currentSimSettings, currentSimResults, model):
    losses, trn_accuracy, val_accuracy, best_phases, best_trf_matrix = currentSimResults
    FOLDER, ONN_Model, loss, phase_uncert, N, dataset_name = currentSimSettings
    if ONN_Model == 'R_P':
        Topology = 'Reck'
    if ONN_Model == 'R_D_P':
        Topology = 'Reck + DMM'
    elif ONN_Model == 'I_P':
        Topology = 'Inverted Reck'
    elif ONN_Model == 'R_I_P':
        Topology = 'Reck + Inverted Reck'
    elif ONN_Model == 'R_D_I_P':
        Topology = 'Reck + DMM + Inverted Reck'
    elif ONN_Model == 'C_Q_P':
        Topology = 'Diamond'
    elif ONN_Model == 'C_W_P':
        Topology = 'Central Diamond'
    else:
        Topology = ONN_Model

    # Plot loss, training acc and val acc
    ax1 = plt.plot()
    plt.plot(losses, color='b')
    plt.xlabel('Epoch')
    plt.ylabel("$\mathcal{L}$", color='b')
    ax2 = plt.gca().twinx()
    ax2.plot(trn_accuracy, color='r')
    ax2.plot(val_accuracy, color='g')
    plt.ylabel('Accuracy', color='r')
    plt.legend(['Training Accuracy', 'Validation Accuracy'])
    plt.title(f'Gradient Descent, Max Validation Accuracy: {max(val_accuracy):.2f}\n Dataset: {dataset_name}, Topology: {Topology}')
    plt.ylim([0, 100])
    plt.savefig(f'{FOLDER}/Figures_Fitting/{ONN_Model}_loss={loss:.3f}dB_uncert={phase_uncert:.3f}Rad_{N}Features.png')
    plt.clf()

    # save a txt file containing the loss, trn acc, val acc, in case i want to replot it using matlab
    df = pd.DataFrame({'Losses':losses, 'Training Accuracy':trn_accuracy, 'Validation Accuracy':val_accuracy})
    df.to_csv(f'{FOLDER}/Data_Fitting/{ONN_Model}_loss={loss:.3f}dB_uncert={phase_uncert:.3f}Rad_{N}Features.txt')

    # Get losses of MZIs
    losses_MZI = model.get_all_losses()
    losses_MZI_flat = [item for sublist in losses_MZI for item in sublist]
    df = pd.DataFrame(losses_MZI_flat, columns=['Losses_MZI_dB'])
    df.to_csv(f'{FOLDER}/Losses_per_MZI/lossPerMZI_{ONN_Model}_loss={loss:.3f}dB_uncert={phase_uncert:.3f}Rad_{N}Features.txt')

    # Save best transformation matrix
    best_trf_matrix = np.array(best_trf_matrix)
    f_obj = open(f'{FOLDER}/Phases/Best_TransformationMatrix_{ONN_Model}_loss={loss:.3f}dB_uncert={phase_uncert:.3f}Rad_{N}Features.txt', 'w')
    f_obj.close()
    with open(f'{FOLDER}/Phases/Best_TransformationMatrix_{ONN_Model}_loss={loss:.3f}dB_uncert={phase_uncert:.3f}Rad_{N}Features.txt', "a") as myfile:
        for trf in best_trf_matrix:
            np.savetxt(myfile, trf, fmt='%.4f%+.4fj, '*len(trf[0]), delimiter=', ')
            myfile.write('\n')

    # Save final transformation matrix
    trf_matrix = np.array(model.get_transformation_matrix())
    f_obj = open(f'{FOLDER}/Phases/Last_TransformationMatrix_{ONN_Model}_loss={loss:.3f}dB_uncert={phase_uncert:.3f}Rad_{N}Features.txt', 'w')
    f_obj.close()
    with open(f'{FOLDER}/Phases/Last_TransformationMatrix_{ONN_Model}_loss={loss:.3f}dB_uncert={phase_uncert:.3f}Rad_{N}Features.txt', "a") as myfile:
        for trf in trf_matrix:
            np.savetxt(myfile, trf, fmt='%.4f%+.4fj, '*len(trf[0]), delimiter=', ')
            myfile.write('\n')

    # Create phase array
    last_phases = model.get_all_phases()
    last_phases_flat = [item for sublist in last_phases for item in sublist]
    df = pd.DataFrame(last_phases_flat, columns=['Theta','Phi'])
    df.to_csv(f'{FOLDER}/Phases/last_phases_{ONN_Model}_loss={loss:.3f}dB_uncert={phase_uncert:.3f}Rad_{N}Features.txt')

    # Save best phases as well
    best_phases_flat = [item for sublist in best_phases for item in sublist]
    df = pd.DataFrame(best_phases_flat, columns=['Theta','Phi'])
    df.to_csv(f'{FOLDER}/Phases/Phases_Best_{ONN_Model}_loss={loss:.3f}dB_uncert={phase_uncert:.3f}Rad_{N}Features.txt')

def saveAccuracyData(FOLDER, currentSimSettings, accuracy):
    FOLDER, ONN_Model, loss, phase_uncert, N, dataset_name = currentSimSettings
    np.savetxt(f"{FOLDER}/acc_{ONN_Model}_loss={loss:.3f}_uncert={phase_uncert:.3f}_{N}Feat.txt", np.array(accuracy).T, delimiter=',', fmt='%.3f')

def saveNonlin(FOLDER, Nonlinearities):
    keys = list(Nonlinearities.keys())
    np.savetxt(FOLDER+'/Nonlinearities.txt', keys, delimiter=" ", fmt="%s")
    for key, activ in Nonlinearities.items():
        x = np.linspace(0.01, 1, 1000)
        plt.plot(x, np.abs(activ.forward_pass(x)), label=key)
        plt.xlabel("Input field (a.u.)")
        plt.ylabel("Output field (a.u.)")
    plt.legend()
    plt.savefig(f'{FOLDER}/Figures/nonlin_activation.png')

def plot_scatter_matrix(X, Y,  figsize=(20, 15)):
    plt.rcParams.update({'font.size': 44})
    df = pd.DataFrame(X)
    df['Labels'] = [np.argmax(y) for y in Y]

    #now plot using pandas
    color_wheel = {0: 'r', 1: 'b', 2: 'g', 3: 'k', 4: 'c', 5: 'm', 6: 'y', 7:'tab:blue', 8:'tab:orange',9:'tab:purple'}
    colors = df["Labels"].map(lambda x: color_wheel.get(x))

    features = ['$X_%d$' % x for x in range(len(X[1]))]

    df = df.rename(columns={v:'$X_%d$' % v for v in range(len(X))})


    axes = scatter_matrix(df[features], alpha=.8, figsize=figsize,
                          diagonal='kde',
                          color=colors, s=100, range_padding=0.1)


    for item in axes:
        for idx, ax in enumerate(item):
            ax.set_yticks([0, 0.5, 1])
            ax.set_xticks([0, 0.5, 1])
            ax.set_yticklabels('')
            ax.set_xticklabels('')

            ax.tick_params(axis='both', which='major', labelsize=24)
            ax.tick_params(axis='both', pad=10)
            ax.xaxis.labelpad = 20
    return axes

if __name__ == '__main__':
    FOLDER = '/home/simon/Downloads'
    N = 4

    sys.path.append('/home/simon/Documents/neuroptica')
    import neuroptica as neu
    import setupSimulation as setSim

    setSim.createFOLDER(FOLDER)

    Nonlinearities = {'a2c0.15_bpReLU2':neu.bpReLU(N, alpha=2, cutoff=0.15), }
    saveNonlin(FOLDER, Nonlinearities)
