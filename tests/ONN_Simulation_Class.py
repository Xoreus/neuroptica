"""
ONN_Simulation_Class.py
ONN Simulation data class, containing all the simulation variables such as loss_dB, phase_uncert, ONN_Setups, X, y, Xt, yt, EPOCHS, ect...
Useful because it makes it possible to pickle.dump() the class into a binary file, retrieving it at a later date to continue testing
saves the Phases of each ONN Setups in the order that they appear

Author: Simon Geoffroy-Gagnon
Edit: 31.01.2020
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from saveSimulationData import plot_scatter_matrix  

class ONN_Simulation:
    def __init__(self):
        " Initialize the ONN simulation class with all the default sim variables"
        self.N = 4
        self.EPOCHS = 500
        self.STEP_SIZE = 0.005
        self.SAMPLES = 1000
        self.DATASET_NUM = 1
        self.ITERATIONS = 200
        self.dataset_name = 'Gauss'

        self.loss_diff = 0.1
        self.loss_dB = np.linspace(0, 3, 31)
        self.loss_dB_test = [0]

        self.phase_uncert = np.linspace(0, 1.5, 16)
        self.phase_uncert_test = [0]

        self.BATCH_SIZE = 2**6
        self.ONN_setup = np.array(['R_D_P', 'C_Q_P'])
        self.topology_name = self.get_topology_name()
        self.Phases = []

        self.ROOT_FOLDER = r'Analysis/'
        self.FUNCTION = r'SingleLossAnalysis/'
        self.FOLDER = '/home/simon/Documents/neuroptica/tests/' + self.ROOT_FOLDER + self.FUNCTION + 'test/'

        self.X = []
        self.y = []
        self.Xt = []
        self.yt = []
        self.phases = []

    def create_dict(self):
        " Creates a dict of the simulation variables"
        simSettings = {'N':self.N, 'EPOCHS':self.EPOCHS, 'STEP_SIZE':self.STEP_SIZE, 'SAMPLES':self.SAMPLES,
                'DATASET_NUM':self.DATASET_NUM, 'ITERATIONS':self.ITERATIONS, 'dataset_name':self.dataset_name, 'loss_diff':self.loss_diff}

        simSettings = pd.DataFrame.from_dict(simSettings, orient='index', columns=['Simulation Settings'])
        return simSettings

    def saveSimSettings(self):
        " save loss_dB, phase_uncert, ITERATIONS, ONN_setups, and N "
        simSettings = self.create_dict()
        simulationSettings = simSettings.to_string()
        output_file = open(f'{self.FOLDER}/SimulationSettings.txt','w')
        output_file.write(simulationSettings)
        output_file.close()

        np.savetxt(f'{self.FOLDER}/loss_dB.txt', self.loss_dB, fmt='%.4f')
        np.savetxt(f'{self.FOLDER}/phase_uncert.txt', self.phase_uncert, fmt='%.4f')
        np.savetxt(f'{self.FOLDER}/phase_uncerts_train.txt', self.phase_uncerts_test, fmt='%.4f')
        np.savetxt(f'{self.FOLDER}/phase_uncerts_test.txt', self.phase_uncerts_test, fmt='%.4f',)

        np.savetxt(f'{self.FOLDER}/ONN_Setups.txt', self.ONN_setup, fmt='%s')

    def get_topology_name(self):
        " Get list of actual topology names instead of C_Q_P"
        topology_name = []
        for ONN_Model in self.ONN_setup:
            if ONN_Model == 'R_P':
                Topo = 'Reck'
            if ONN_Model == 'R_D_P':
                Topo = 'Reck + DMM'
            if ONN_Model == 'R_D_P':
                Topo = 'Reck + DMM'
            elif ONN_Model == 'I_P':
                Topo = 'Inverted Reck'
            elif ONN_Model == 'R_I_P':
                Topo = 'Reck + Inverted Reck'
            elif ONN_Model == 'R_D_I_P':
                Topo = 'Reck + DMM + Inverted Reck'
            elif ONN_Model == 'C_Q_P':
                Topo = 'Diamond'
            elif ONN_Model == 'C_W_P':
                Topo = 'Central Diamond'
            elif ONN_Model == 'E_P':
                Topo = 'Clements'
            else:
                Topo = ONN_Model
            topology_name.append(Topo)
        return topology_name

    def saveSimDataset(self):
        " Save simulation's datasset, both in plot and txt form"
        axes = plot_scatter_matrix(self.X, self.y)
        plt.savefig(f'{self.FOLDER}/Datasets/{self.dataset_name}_Samples={len(self.X)}_Dataset.png')
        plt.clf()
        mpl.rcParams.update(mpl.rcParamsDefault)

        np.savetxt(f'{self.FOLDER}/Datasets/{self.dataset_name}_X_{self.N}Features_{len(self.y[0])}Classes_Samples={len(self.X)}_Dataset.txt', self.X, delimiter=',',fmt='%.3f')
        np.savetxt(f'{self.FOLDER}/Datasets/{self.dataset_name}_y_{self.N}Features_{len(self.y[0])}Classes_Samples={len(self.X)}_Dataset.txt', self.y, delimiter=',',fmt='%.3f')
        np.savetxt(f'{self.FOLDER}/Datasets/{self.dataset_name}_Xt_{self.N}Features_{len(self.y[0])}Classes_Samples={len(self.X)}_Dataset.txt', self.Xt, delimiter=',',fmt='%.3f')
        np.savetxt(f'{self.FOLDER}/Datasets/{self.dataset_name}_yt_{self.N}Features_{len(self.y[0])}Classes_Samples={len(self.X)}_Dataset.txt', self.yt, delimiter=',',fmt='%.3f')

    def normalize_dataset(self):
        " Normalize the dataset to be in range [0, 1]"
        self.X = (self.X - np.min(self.X))/(np.max(self.X) - np.min(self.X))
        self.Xt = (self.Xt - np.min(self.Xt))/(np.max(self.Xt) - np.min(self.Xt))
        return self.X, self.y, self.Xt, self.yt
