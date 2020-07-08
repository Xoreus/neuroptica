"""
ONN_Simulation_Class.py
ONN Simulation data class, containing all the simulation variables such as loss_dB, phase_uncert, ONN_Setups, X, y, Xt, yt, EPOCHS, ect...
Useful because it makes it possible to pickle.dump() the class into a binary file, retrieving it at a later date to continue testing
saves the Phases of each ONN Setups in the order that they appear
also saves the 3D accuracy array to a .mat file with accuracy[losses_dB, theta, phi]

Author: Simon Geoffroy-Gagnon
Edit: 2020.07.07
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import scipy.io
import pickle
import matplotlib as mpl

class ONN_Simulation:
    def __init__(self):
        " Initialize the ONN simulation class with all the default sim variables"
        # Training and Testing
        self.SAMPLES = 1000
        self.dataset_name = 'Gauss'
        self.N = 4
        self.BATCH_SIZE = 2**6
        self.EPOCHS = 600
        self.STEP_SIZE = 0.0005
        self.ITERATIONS = 40 # number of times to retry same loss/PhaseUncert
        self.loss_diff = 0 # \sigma dB
        self.loss_dB = np.linspace(0, 2, 11)
        self.phase_uncert_theta = np.linspace(0., 2.5, 21)
        self.phase_uncert_phi = np.linspace(0., 2.5, 21)
        self.same_phase_uncert = False
        self.rng = 1
        self.zeta = 0.75

        self.loss_dB = np.linspace(0, 3, 31)
        self.loss_dB_test = [0]

        self.BATCH_SIZE = 2**6
        self.topo = 'R_P'

        self.FOLDER = 'Analysis'

        # Datasets
        self.X = []
        self.y = []
        self.Xt = []
        self.yt = []
        self.range_dB = 5
        
        # Phases and transformation Matrices
        self.phases = []
        self.trf_matrix = [] 

        # Training and Simulation results
        self.losses = []
        self.val_accuracy = []
        self.trn_accuracy = []
        self.accuracy = []
        self.accuracy_LPU = []
        self.accuracy_PT = []
    def get_topology_name(self):
        " Get list of actual topology names instead of C_Q_P"
        if self.topo == 'R_D_P':
            Topo = 'Reck + DMM'
        elif self.topo == 'I_P':
            Topo = 'Inverted Reck'
        elif self.topo == 'R_I_P':
            Topo = 'Reck + Inverted Reck'
        elif self.topo == 'R_D_I_P':
            Topo = 'Reck + DMM + Inverted Reck'
        elif self.topo == 'C_Q_P':
            Topo = 'Diamond'
        elif self.topo == 'C_W_P':
            Topo = 'Central Diamond'
        elif self.topo == 'E_P':
            Topo = 'Clements'
        elif self.topo == 'R_P':
            Topo = 'Reck'
        else:
            Topo = self.topo.replace('_','')
        self.topology = Topo
    def saveSimSettings(self):
        " save loss_dB, phase_uncert, ITERATIONS, ONN_setups, and N "
        simSettings = self.create_dict()
        simulationSettings = simSettings.to_string()
        with open(f'{self.FOLDER}/SimulationSettings.txt','w') as f:
            f.write(simulationSettings)
    def saveSimData(self, model):
        ''' simulation data to .txt files'''
        df = pd.DataFrame({'Losses':self.losses, 'Training Accuracy':self.trn_accuracy, 'Validation Accuracy':self.val_accuracy})
        df.to_csv(f'{self.FOLDER}/backpropagation-{self.topo}.txt')

        # Save best transformation matrix
        best_trf_matrix = np.array(self.best_trf_matrix)
        with open(f'{self.FOLDER}/TransformationMatrix_{self.topo}_.txt', "w") as myfile:
            for trf in best_trf_matrix:
                np.savetxt(myfile, trf, fmt='%.4f%+.4fj, '*len(trf[0]), delimiter=', ')
                myfile.write('\n')

        # Save best phases as well
        best_phases_flat = [item for sublist in self.phases for item in sublist]
        df = pd.DataFrame(best_phases_flat, columns=['Theta','Phi'])
        df.to_csv(f'{self.FOLDER}/Phases_Best_{self.topo}.txt')
    def saveAccuracyData(self):
        ''' save the accuracy computed from calculate_accuracy '''
        scipy.io.savemat(f"{self.FOLDER}/acc_{self.topo}_N={self.N}.mat", mdict={'accuracy':self.accuracy})
    def saveSimDataset(self):
        ''' Save simulation's datasset, both in plot and txt form '''
        np.savetxt(f'{self.FOLDER}/Datasets/y.txt', self.y, delimiter=',',fmt='%.3f')
        np.savetxt(f'{self.FOLDER}/Datasets/Xt.txt', self.Xt, delimiter=',',fmt='%.3f')
        np.savetxt(f'{self.FOLDER}/Datasets/yt.txt', self.yt, delimiter=',',fmt='%.3f')
        np.savetxt(f'{self.FOLDER}/Datasets/X.txt', self.X, delimiter=',',fmt='%.3f')

        # Normalize output in case this was not done
        Xn = (self.X - np.min(self.X))/(np.max(self.X) - np.min(self.X))
        Xtn = (self.Xt - np.min(self.Xt))/(np.max(self.Xt) - np.min(self.Xt))
        np.savetxt(f'{self.FOLDER}/Datasets/X_normalized.txt', Xn, delimiter=',',fmt='%.3f')
        np.savetxt(f'{self.FOLDER}/Datasets/Xt_normalized.txt', Xtn, delimiter=',',fmt='%.3f')

        # Also output the power of the input vectors to help with experimental testing
        np.savetxt(f'{self.FOLDER}/Datasets/X_Power.txt', np.abs(self.X)**2, delimiter=',',fmt='%.3f') 
        np.savetxt(f'{self.FOLDER}/Datasets/Xt_Power.txt', np.abs(self.X)**2, delimiter=',',fmt='%.3f')

        # Also save the power of the normalized input samples in dB, normalized with range [-range_dB 0] dB. Required to take the second min, since first min is 0 or -inf dB.
        Xn = 10*np.log10(np.abs(Xn)**2+sorted(set(np.abs(Xn).reshape(-1)))[1])
        Xtn = 10*np.log10(np.abs(Xtn)**2+sorted(set(np.abs(Xtn).reshape(-1)))[1])
        Xn = ((Xn - np.min(Xn))/(np.max(Xn) - np.min(Xn)) - 1)*self.range_dB
        Xtn = ((Xtn - np.min(Xtn))/(np.max(Xtn) - np.min(Xtn)) - 1)*self.range_dB

        np.savetxt(f'{self.FOLDER}/Datasets/X_Power_normalized_[-{self.range_dB}-0]dB.txt', Xn, delimiter=',',fmt='%.3f') 
        np.savetxt(f'{self.FOLDER}/Datasets/Xt_Power_normalized_[-{self.range_dB}-0]dB.txt', Xtn, delimiter=',',fmt='%.3f')
    def create_dict(self):
        " Creates a dict of the simulation variables"
        simSettings = {'N':self.N, 'EPOCHS':self.EPOCHS, 'STEP_SIZE':self.STEP_SIZE, 'SAMPLES':self.SAMPLES,
                'DATASET_NUM':self.DATASET_NUM, 'ITERATIONS':self.ITERATIONS, 'dataset_name':self.dataset_name, 'loss_diff':self.loss_diff}

        simSettings = pd.DataFrame.from_dict(simSettings, orient='index', columns=['Simulation Settings'])
        return simSettings
    def saveSelf(self):
        ''' save .mat strutcure of this class' variables '''
        scipy.io.savemat(f"{self.FOLDER}/{self.topo}.mat", mdict={f'{self.topo}':self})
    def saveAll(self, model):
        self.saveSimDataset()
        self.saveSimData(model)
        self.plotAll()
        self.saveSelf()
    def plotAll(self):
        fig, ax1 = plt.subplots()
        color = 'tab:blue'
        ax1.set_xlabel('Epochs')
        ax1.set_ylabel('$\mathcal{L}$', color=color)
        lns1 = ax1.plot(self.losses, color='tab:blue', label='Losses')
        ax1.tick_params(axis='y', labelcolor=color)
        ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
        color = 'tab:red'
        ax2.set_ylabel('Accuracy (\%)', color=color)  # we already handled the x-label with ax1
        ax2.set_ylim([0, 100])
        lns2 = ax2.plot(self.val_accuracy, color='tab:red', label='Validation Accuracy')
        lns3 = ax2.plot(self.trn_accuracy, color='k', label='Training Accuracy')
        lns = lns1+lns2+lns3
        labs = [l.get_label() for l in lns]
        ax1.legend(lns, labs, loc=0)
        ax2.tick_params(axis='y', labelcolor=color)
        fig.tight_layout() 
        # plt.legend(['Validation Accuracy','Training Accuracy'])
        plt.savefig(f'{self.FOLDER}/backprop_{self.topo}.pdf')
        plt.clf()

        plt.pcolor(self.accuracy_LPU, vmin=100/(self.N+1), vmax=100, cmap='magma', rasterized=True)
        plt.xlabel('Loss/MZI (dB)')
        plt.ylabel(r'$\sigma_{\theta}, \sigma_\phi$ (Rad)')
        cbar = plt.colorbar()
        cbar.set_label('Accuracy (\%)')
        plt.savefig(f'{self.FOLDER}/LPU_ACC_{self.topo}.pdf')
        plt.clf()

        plt.pcolor(self.accuracy_PT, vmin=100/(self.N+1), vmax=100, cmap='magma', rasterized=True)
        plt.xlabel(r'$\sigma_\theta$ (Rad)')
        plt.ylabel(r'$\sigma_{\phi}$ (Rad)')
        cbar = plt.colorbar()
        cbar.set_label('Accuracy (\%)')
        plt.savefig(f'{self.FOLDER}/PT_ACC_{self.topo}.pdf')
        plt.clf()
    def createFOLDER(self):
        if not os.path.isdir(self.FOLDER):
            os.makedirs(self.FOLDER)
        if not os.path.isdir(self.FOLDER + '/Datasets'):
            os.makedirs(self.FOLDER + '/Datasets')
    def pickle_save(self):
        with open(f'{self.FOLDER}/{self.topo}.pkl', 'wb') as p:
            pickle.dump(self, p)
    def pickle_load(self):
        with open(f'{self.FOLDER}/{self.topo}.pkl', 'rb') as p:
            self = pickle.load(p)
            return self 
