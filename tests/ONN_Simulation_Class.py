"""
ONN_Simulation_Class.py
ONN Simulation data class, containing all the simulation variables such as loss_dB, phase_uncert, ONN_Setups, X, y, Xt, yt, EPOCHS, ect...
Useful because it makes it possible to pickle.dump() the class into a binary file, retrieving it at a later date to continue testing
saves the Phases of each ONN Setups in the order that they appear
also saves the 3D accuracy array to a .mat file with accuracy[losses_dB, theta, phi]

Author: Simon Geoffroy-Gagnon
Edit: 05.02.2020
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.io
import matplotlib as mpl
mpl.use('Agg')
from saveSimulationData import plot_scatter_matrix  

class ONN_Simulation:
    def __init__(self):
        " Initialize the ONN simulation class with all the default sim variables"
        self.N = 4
        self.EPOCHS = 500
        self.STEP_SIZE = 0.005
        self.SAMPLES = 1000
        self.ITERATIONS = 200
        self.dataset_name = 'Gauss'

        self.loss_diff = 0.1
        self.loss_dB = np.linspace(0, 3, 31)
        self.loss_dB_test = [0]

        self.phase_uncert_theta = np.linspace(0, 1.5, 16)
        self.phase_uncert_phi = np.linspace(0, 1.5, 16)

        self.BATCH_SIZE = 2**6
        self.ONN_setup = np.array(['R_D_P', 'C_Q_P'])
        self.onn_topo = 'allTopologies'

        self.ROOT_FOLDER = r'Analysis/'
        self.FUNCTION = r'SingleLossAnalysis/'
        self.FOLDER = '/home/simon/Documents/neuroptica/tests/' + self.ROOT_FOLDER + self.FUNCTION

        self.X = []
        self.y = []
        self.Xt = []
        self.yt = []

        self.phases = []
        self.trf_matrix = [] 

        self.losses = []
        self.val_accuracy = []
        self.trn_accuracy = []
        self.accuracy = []
    def normalize_dataset(self):
        " Normalize the dataset to be in range [0, 1]"
        self.X = (self.X - np.min(self.X))/(np.max(self.X) - np.min(self.X))
        self.Xt = (self.Xt - np.min(self.Xt))/(np.max(self.Xt) - np.min(self.Xt))
        return np.array(self.X), np.array(self.y), np.array(self.Xt), np.array(self.yt)
    def create_dict(self):
        " Creates a dict of the simulation variables"
        simSettings = {'N':self.N, 'EPOCHS':self.EPOCHS, 'STEP_SIZE':self.STEP_SIZE, 'SAMPLES':self.SAMPLES,
                'DATASET_NUM':self.DATASET_NUM, 'ITERATIONS':self.ITERATIONS, 'dataset_name':self.dataset_name, 'loss_diff':self.loss_diff}

        simSettings = pd.DataFrame.from_dict(simSettings, orient='index', columns=['Simulation Settings'])
        return simSettings
    def get_topology_name(self):
        " Get list of actual topology names instead of C_Q_P"

        if self.onn_topo == 'R_D_P':
            Topo = 'Reck + DMM'
        if self.onn_topo == 'R_D_P':
            Topo = 'Reck + DMM'
        elif self.onn_topo == 'I_P':
            Topo = 'Inverted Reck'
        elif self.onn_topo == 'R_I_P':
            Topo = 'Reck + Inverted Reck'
        elif self.onn_topo == 'R_D_I_P':
            Topo = 'Reck + DMM + Inverted Reck'
        elif self.onn_topo == 'C_Q_P':
            Topo = 'Diamond'
        elif self.onn_topo == 'C_W_P':
            Topo = 'Central Diamond'
        elif self.onn_topo == 'E_P':
            Topo = 'Clements'
        elif self.onn_topo == 'R_P':
            Topo = 'Reck'
        self.topology = Topo
    def get_all_topologies(self):
        " Get list of actual topology names instead of C_Q_P"
        models = []
        for model in self.ONN_setup:
            if model == 'R_P':
                Topo = 'Reck'
            if model == 'R_D_P':
                Topo = 'Reck + DMM'
            if model == 'R_D_P':
                Topo = 'Reck + DMM'
            elif model == 'I_P':
                Topo = 'Inverted Reck'
            elif model == 'R_I_P':
                Topo = 'Reck + Inverted Reck'
            elif model == 'R_D_I_P':
                Topo = 'Reck + DMM + Inverted Reck'
            elif model == 'C_Q_P':
                Topo = 'Diamond'
            elif model == 'C_W_P':
                Topo = 'Central Diamond'
            elif model == 'E_P':
                Topo = 'Clements'
            else:
                Topo = self.onn_topo
            models.append(Topo)
        self.models = models
    def saveSimSettings(self):
        " save loss_dB, phase_uncert, ITERATIONS, ONN_setups, and N "
        simSettings = self.create_dict()
        simulationSettings = simSettings.to_string()
        with open(f'{self.FOLDER}/SimulationSettings.txt','w') as f:
            f.write(simulationSettings)

        np.savetxt(f'{self.FOLDER}/loss_dB.txt', self.loss_dB, fmt='%.4f')

        np.savetxt(f'{self.FOLDER}/phase_uncert_theta.txt', self.phase_uncert_theta, fmt='%.4f')
        np.savetxt(f'{self.FOLDER}/phase_uncert_phi.txt', self.phase_uncert_phi, fmt='%.4f')

        np.savetxt(f'{self.FOLDER}/ONN_Setups.txt', self.ONN_setup, fmt='%s')
    def saveSelf(self):
        ''' save .mat strutcure of this class' variables '''
        scipy.io.savemat(f"{self.FOLDER}/{self.onn_topo}.mat", mdict={f'{self.onn_topo}':self})
    def saveSimData(self, model):
        ''' Plot loss, training acc and val acc '''
        ax1 = plt.plot()
        plt.plot(self.losses, color='b')
        plt.xlabel('Epoch')
        plt.ylabel("$\mathcal{L}$", color='b')
        ax2 = plt.gca().twinx()
        ax2.plot(self.trn_accuracy, color='r')
        ax2.plot(self.val_accuracy, color='g')
        plt.ylabel('Accuracy', color='r')
        plt.legend(['Training Accuracy', 'Validation Accuracy'])
        plt.title(f'Gradient Descent, Max Validation Accuracy: {max(self.val_accuracy):.2f}\n Dataset: {self.dataset_name}, Topology: {self.Topology}')
        plt.ylim([0, 100])
        plt.savefig(f'{self.FOLDER}/Figures_Fitting/{self.onn_topo}_loss={self.loss_dB[0]:.3f}dB_uncert={self.phase_uncert:.3f}Rad_{self.N}Features.png')
        plt.clf()

        # Get losses of MZIs
        losses_MZI = model.get_all_losses()
        losses_MZI_flat = [item for sublist in losses_MZI for item in sublist]
        df = pd.DataFrame(losses_MZI_flat, columns=['Losses_MZI_dB'])
        df.to_csv(f'{self.FOLDER}/Losses_per_MZI/lossPerMZI_{self.onn_topo}_loss={self.loss_dB[0]:.3f}dB_uncert={self.phase_uncert:.3f}Rad_{self.N}Features.txt')

        # Save final transformation matrix
        onn.last_trf_matrix = np.array(model.get_transformation_matrix())

        # Create phase array
        onn.last_phases = model.get_all_phases()
    def saveAccuracyData(self):
        ''' save the accuracy computed from calculate_accuracy '''
        scipy.io.savemat(f"{self.FOLDER}/acc_{self.onn_topo}_loss={self.loss_dB[0]:.3f}_uncert={self.phase_uncert_theta[0]:.3f}_{self.N}Feat.mat", mdict={'accuracy':self.accuracy})
    def saveSimDataset(self):
        ''' Save simulation's datasset, both in plot and txt form '''
        axes = plot_scatter_matrix(self.X, self.y)
        plt.savefig(f'{self.FOLDER}/Datasets/{self.dataset_name}_Samples={len(self.X)}_Dataset.png')
        plt.clf()
        mpl.rcParams.update(mpl.rcParamsDefault)
        np.savetxt(f'{self.FOLDER}/Datasets/{self.dataset_name}_y_{self.N}Features_{len(self.y[0])}Classes_Samples={len(self.X)}_Dataset.txt', self.y, delimiter=',',fmt='%.3f')
        np.savetxt(f'{self.FOLDER}/Datasets/{self.dataset_name}_Xt_{self.N}Features_{len(self.y[0])}Classes_Samples={len(self.X)}_Dataset.txt', self.Xt, delimiter=',',fmt='%.3f')
        np.savetxt(f'{self.FOLDER}/Datasets/{self.dataset_name}_yt_{self.N}Features_{len(self.y[0])}Classes_Samples={len(self.X)}_Dataset.txt', self.yt, delimiter=',',fmt='%.3f')

if __name__ == '__main__':
    ONN = ONN_Simulation()
    ONN.FOLDER = '/home/simon/Desktop/'
    ONN.saveSelf()
