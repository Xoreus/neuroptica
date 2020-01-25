""" 
Function to save all of the simulation data at every loss/phase_uncert training setting

Author: Simon Geoffroy-Gagnon
Edit: 17.01.2020
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def saveSimSettings(FOLDER, simSettings):
    " save loss_dB, phase_uncert, ITERATIONS, ONN_setups, and N "
    N, EPOCHS, STEP_SIZE, SAMPLES, DATASET_NUM, ITERATIONS, losses_dB, phase_uncerts, dataset_name, ONN_setups, loss_diff = simSettings 

    np.savetxt(f'{FOLDER}/N.txt', [N], fmt='%d')
    np.savetxt(f'{FOLDER}/EPOCHS.txt', [EPOCHS], fmt='%.3f')
    np.savetxt(f'{FOLDER}/STEP_SIZE.txt', [STEP_SIZE], fmt='%.3f')
    np.savetxt(f'{FOLDER}/SAMPLES.txt', [SAMPLES], fmt='%.3f')
    np.savetxt(f'{FOLDER}/DATASET_NUM.txt', [DATASET_NUM], fmt='%.3f')
    np.savetxt(f'{FOLDER}/ITERATIONS.txt', [ITERATIONS], fmt='%.3f')
    np.savetxt(f'{FOLDER}/LossdB.txt', losses_dB, delimiter=',', fmt='%.3f')
    np.savetxt(f'{FOLDER}/PhaseUncert.txt', phase_uncerts, delimiter=',', fmt='%.3f')
    np.savetxt(f'{FOLDER}/dataset_name.txt', [ITERATIONS], delimiter=',', fmt='%.d')
    np.savetxt(f'{FOLDER}/ONN_Setups.txt', [setup for setup in ONN_setups], delimiter=" ", fmt="%s")
    np.savetxt(f'{FOLDER}/loss_diff.txt', [loss_diff], fmt="%s")

def saveSimSettings_multiTrainings(FOLDER, simSettings):
    " save loss_dB, phase_uncert, ITERATIONS, ONN_setups, and N "
    N, EPOCHS, STEP_SIZE, SAMPLES, DATASET_NUM, ITERATIONS, losses_dB_train, losses_dB_test, phase_uncerts_train, phase_uncerts_test, dataset_name, ONN_setups = simSettings

    np.savetxt(f'{FOLDER}/N.txt', [N], fmt='%d')
    np.savetxt(f'{FOLDER}/EPOCHS.txt', [EPOCHS], fmt='%.3f')
    np.savetxt(f'{FOLDER}/STEP_SIZE.txt', [STEP_SIZE], fmt='%.3f')
    np.savetxt(f'{FOLDER}/SAMPLES.txt', [SAMPLES], fmt='%.3f')
    np.savetxt(f'{FOLDER}/DATASET_NUM.txt', [DATASET_NUM], fmt='%.3f')
    np.savetxt(f'{FOLDER}/ITERATIONS.txt', [ITERATIONS], fmt='%.3f')
    np.savetxt(f'{FOLDER}/LossdB_train.txt', losses_dB_train, delimiter=',', fmt='%.3f')
    np.savetxt(f'{FOLDER}/LossdB_test.txt', losses_dB_test, delimiter=',', fmt='%.3f')
    np.savetxt(f'{FOLDER}/PhaseUncert_train.txt', phase_uncerts_train, delimiter=',', fmt='%.3f')
    np.savetxt(f'{FOLDER}/PhaseUncert_test.txt', phase_uncerts_test, delimiter=',', fmt='%.3f')
    np.savetxt(f'{FOLDER}/dataset_name.txt', [ITERATIONS], delimiter=',', fmt='%.d')
    np.savetxt(f'{FOLDER}/ONN_Setups.txt', [setup for setup in ONN_setups], delimiter=" ", fmt="%s")

def saveSimData(currentSimSettings, currentSimResults, model):
    losses, trn_accuracy, val_accuracy, best_phases, best_trf_matrix = currentSimResults
    FOLDER, ONN_Model, loss, phase_uncert, N, ii, NonLin_key, dataset_name = currentSimSettings
    if ONN_Model == 'R_P':
        Topology = 'Reck'
    elif ONN_Model == 'I_P':
        Topology = 'Inverted Reck'
    elif ONN_Model == 'R_I_P':
        Topology = 'Reck + Inverted Reck'
    elif ONN_Model == 'R_D_I_P':
        Topology = 'Reck + DMM + Inverted Reck'
    elif ONN_Model == 'C_Q_P':
        Topology = 'Diamond'
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
    plt.savefig(f'{FOLDER}/Figures_Fitting/{ONN_Model}_loss={loss:.3f}dB_uncert={phase_uncert:.3f}Rad_{N}Features_#{ii}_{NonLin_key}.png')
    plt.clf()

    # save a txt file containing the loss, trn acc, val acc, in case i want to replot it using matlab
    df = pd.DataFrame({'Losses':losses, 'Training Accuracy':trn_accuracy, 'Validation Accuracy':val_accuracy})
    df.to_csv(f'{FOLDER}/Data_Fitting/{ONN_Model}_loss={loss:.3f}dB_uncert={phase_uncert:.3f}Rad_{N}Features_#{ii}_{NonLin_key}.txt')

    # Get losses of MZIs
    losses_MZI = model.get_all_losses()
    losses_MZI_flat = [item for sublist in losses_MZI for item in sublist]
    df = pd.DataFrame(losses_MZI_flat, columns=['Losses_MZI_dB'])
    df.to_csv(f'{FOLDER}/Losses_per_MZI/lossPerMZI_{ONN_Model}_loss={loss:.3f}dB_uncert={phase_uncert:.3f}Rad_{N}Features_#{ii}_{NonLin_key}.txt')

    # Save best transformation matrix
    best_trf_matrix = np.array(best_trf_matrix)
    f_obj = open(f'{FOLDER}/Phases/Best_TransformationMatrix_{ONN_Model}_loss={loss:.3f}dB_uncert={phase_uncert:.3f}Rad_{N}Features_#{ii}_{NonLin_key}.txt', 'w')
    f_obj.close()
    with open(f'{FOLDER}/Phases/Best_TransformationMatrix_{ONN_Model}_loss={loss:.3f}dB_uncert={phase_uncert:.3f}Rad_{N}Features_#{ii}_{NonLin_key}.txt', "a") as myfile:
        for trf in best_trf_matrix:
            np.savetxt(myfile, trf, fmt='%.4f%+.4fj, '*len(trf[0]), delimiter=', ')
            myfile.write('\n')

    # Save final transformation matrix
    trf_matrix = np.array(model.get_transformation_matrix())
    f_obj = open(f'{FOLDER}/Phases/Last_TransformationMatrix_{ONN_Model}_loss={loss:.3f}dB_uncert={phase_uncert:.3f}Rad_{N}Features_#{ii}_{NonLin_key}.txt', 'w')
    f_obj.close()
    with open(f'{FOLDER}/Phases/Last_TransformationMatrix_{ONN_Model}_loss={loss:.3f}dB_uncert={phase_uncert:.3f}Rad_{N}Features_#{ii}_{NonLin_key}.txt', "a") as myfile:
        for trf in trf_matrix:
            np.savetxt(myfile, trf, fmt='%.4f%+.4fj, '*len(trf[0]), delimiter=', ')
            myfile.write('\n')

    # Create phase array
    last_phases = model.get_all_phases()
    last_phases_flat = [item for sublist in last_phases for item in sublist]
    df = pd.DataFrame(last_phases_flat, columns=['Theta','Phi'])
    df.to_csv(f'{FOLDER}/Phases/last_phases_{ONN_Model}_loss={loss:.3f}dB_uncert={phase_uncert:.3f}Rad_{N}Features_#{ii}_{NonLin_key}.txt')

    # Save best phases as well
    best_phases_flat = [item for sublist in best_phases for item in sublist]
    df = pd.DataFrame(best_phases_flat, columns=['Theta','Phi'])
    df.to_csv(f'{FOLDER}/Phases/Phases_Best_{ONN_Model}_loss={loss:.3f}dB_uncert={phase_uncert:.3f}Rad_{N}Features_#{ii}_{NonLin_key}.txt')

def saveAccuracyData(FOLDER, currentSimSettings, accuracy):
    FOLDER, ONN_Model, loss, phase_uncert, N, ii, NonLin_key, dataset_name = currentSimSettings
    np.savetxt(f"{FOLDER}/acc_{ONN_Model}_loss={loss:.3f}_uncert={phase_uncert:.3f}_{N}Feat_{NonLin_key}_set{ii}.txt", np.array(accuracy).T, delimiter=',', fmt='%.3f')

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
