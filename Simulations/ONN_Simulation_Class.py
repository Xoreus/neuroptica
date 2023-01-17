"""
ONN_Simulation_Class.py
ONN Simulation data class, containing all the simulation variables such as loss_dB, phase_uncert, ONN_Setups, X, y, Xt, yt, EPOCHS, ect...
Useful because it makes it possible to pickle.dump() the class into a binary file, retrieving it at a later date to continue testing
saves the Phases of each ONN Setups in the order that they appear
also saves the 3D accuracy array to a .mat file with accuracy[losses_dB, theta, phi]

Author: Simon Geoffroy-Gagnon
Edit: 2020.07.07

Edit: 2022.10.13 by Bokun Zhao (bokun.zhao@mail.mcgill.ca)
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_filter
import os
from cm_maker import plot_confusion_matrix
import scipy.io
import pickle
import matplotlib as mpl
from matplotlib.ticker import FormatStrFormatter
import xlsxwriter
import io
from matplotlib import rc
np.seterr(divide = 'ignore') 

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
        self.zeta = 0.1

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
        
        # Model, Phases and transformation Matrices
        self.model = []
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
        elif self.topo == 'B_C_Q_P':
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
        with open(f'{self.FOLDER}/TransformationMatrix_{self.topo}.txt', "w") as f:
            for trf in best_trf_matrix:
                np.savetxt(f, trf, fmt='%.4f%+.4fj, '*len(trf[0]), delimiter=', ')
                f.write('\n')

        # Save best phases as well
        # print(self.phases)
        best_phases_flat = [item for sublist in self.phases for item in sublist]
        theta = [ph[0] for ph in best_phases_flat]
        phi = [ph[1] for ph in best_phases_flat]
        # print(f'\n'.join('{:.9f}'.format(k) for k in theta))
        df = pd.DataFrame(best_phases_flat, columns=['Theta','Phi'])
        with open(f'{self.FOLDER}/best_phases_matlab_{self.topo}.txt', "w") as f:
            f.write("Theta=[\n")
            f.write(f'\n'.join('{:.9f}'.format(k) for k in theta))
            f.write("];\n\nPhi=[\n")
            f.write(f'\n'.join('{:.9f}'.format(k) for k in phi))
            f.write("];")
        df.to_csv(f'{self.FOLDER}/best_phases_highPrecision_{self.topo}.txt')
    def saveAccuracyData(self):
        ''' save the accuracy computed from calculate_accuracy '''
        scipy.io.savemat(f"{self.FOLDER}/acc_{self.topo}_N={self.N}.mat", mdict={'accuracy':self.accuracy})
    def saveSimDataset(self):
        ''' Save simulation's datasset, both in plot and txt form '''
        np.savetxt(f'{self.FOLDER}/Datasets/y.txt', self.y, delimiter=',',fmt='%d')
        np.savetxt(f'{self.FOLDER}/Datasets/Xt.txt', self.Xt, delimiter=',',fmt='%.6f')
        np.savetxt(f'{self.FOLDER}/Datasets/yt.txt', self.yt, delimiter=',',fmt='%d')
        np.savetxt(f'{self.FOLDER}/Datasets/X.txt', self.X, delimiter=',',fmt='%.6f')

        # Also output the power of the input vectors to help with experimental testing
        np.savetxt(f'{self.FOLDER}/Datasets/X_Power.txt', np.abs(self.X)**2, delimiter=',',fmt='%.6f') 
        np.savetxt(f'{self.FOLDER}/Datasets/Xt_Power.txt', np.abs(self.Xt)**2, delimiter=',',fmt='%.6f')
        
        np.savetxt(f'{self.FOLDER}/Datasets/X_Power_dB.txt', 10*np.log10(np.abs(self.X)**2), delimiter=',',fmt='%.6f') 
        np.savetxt(f'{self.FOLDER}/Datasets/Xt_Power_dB.txt', 10*np.log10(np.abs(self.Xt)**2), delimiter=',',fmt='%.6f')
    def save_correct_classified_samples(self, model, zeta=0):
        ''' Save only the correct validation samples for a dataset
        Uses the zeta parameter from Shokraneh et al's paper in 2019'''
        Y_hat = model.forward_pass(self.Xt.T)
        pred = np.array([np.argmax(yhat) for yhat in Y_hat.T])
        gt = np.array([np.argmax(tru) for tru in self.yt])

        sorted_y_hat = [sorted(Y) for Y in Y_hat.T]    
        diff_gt_zeta = [True if Y[-1] - Y[-2] > zeta else False for Y in sorted_y_hat]

        correct_class = pred == gt 
        corr = [diff_gt_zeta[idx]*correct_class[idx] for idx, _ in enumerate(correct_class)]

        ax = plot_confusion_matrix(pred, gt, list(range(self.classes)),
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues)
        bottom, top = ax.get_ylim()
        # ax.set_ylim(bottom + 0.5, top - 0.5)
        plt.tight_layout()
        plt.savefig(f'{self.FOLDER}/cm.pdf')

        Xt_correct = self.Xt[corr]
        yt_correct = self.yt[corr]
        np.savetxt(f'{self.FOLDER}/Datasets/YHat_t_correct_{zeta:.2f}W_difference.txt', Y_hat.T, delimiter=', ', fmt='%.4f')

        Y_hat = model.forward_pass(self.X.T)
        pred = np.array([np.argmax(yhat) for yhat in Y_hat.T])
        gt = np.array([np.argmax(tru) for tru in self.y])

        sorted_y_hat = [sorted(Y) for Y in Y_hat.T]    
        diff_gt_zeta = [True if Y[-1] - Y[-2] > zeta else False for Y in sorted_y_hat]

        correct_class = pred == gt 
        # print(correct_class)
        # print(diff_gt_zeta)
        corr = [diff_gt_zeta[idx]*correct_class[idx] for idx, _ in enumerate(correct_class)]
        # print(corr)
        # print(f"sum of correct classifications: {sum(corr)}")

        X_correct = self.X[corr] 
        y_correct = self.y[corr] 
        Y = np.vstack([yt_correct, y_correct])

        print(f"Correct Classes Total (zeta = {zeta} W): {np.sum(Y, axis=0)}, Validation Accuracy: {np.sum(yt_correct)/(len(self.yt))*100:.2f}%")
        np.savetxt(f"{self.FOLDER}/Datasets/X_correct_{zeta:.2f}W_difference.txt", np.vstack([Xt_correct, X_correct]), fmt='%.3f', delimiter=', ')
        np.savetxt(f"{self.FOLDER}/Datasets/X_correct_power_dB_{zeta:.2f}W_difference.txt", np.vstack([10*np.log10(Xt_correct**2), 10*np.log10(X_correct**2)]), fmt='%.3f', delimiter=', ')
        np.savetxt(f"{self.FOLDER}/Datasets/y_correct_{zeta:.2f}W_difference.txt", Y, fmt='%d', delimiter=', ')
    def create_dict(self):
        " Creates a dict of the simulation variables"
        simSettings = {'N':self.N, 'EPOCHS':self.EPOCHS, 'STEP_SIZE':self.STEP_SIZE, 'SAMPLES':self.SAMPLES,
                'DATASET_NUM':self.DATASET_NUM, 'ITERATIONS':self.ITERATIONS, 'dataset_name':self.dataset_name, 'loss_diff':self.loss_diff}

        simSettings = pd.DataFrame.from_dict(simSettings, orient='index', columns=['Simulation Settings'])
        return simSettings
    def saveSelf(self):
        ''' save .mat strutcure of this class' variables '''
        model = self.model
        self.model = []
        self_dict = {f'{self.topo}':self}
        scipy.io.savemat(f"{self.FOLDER}/Topologies/{self.topo}.mat", mdict={f'{self.topo}':self})
        self.model = model
    def saveForwardPropagation(self, model):
        y_hat = model.forward_pass(self.Xt.T)
        np.savetxt(f'{self.FOLDER}/Datasets/yhat_t_power.txt', y_hat.T, fmt='%.5f', delimiter=', ')
        np.savetxt(f'{self.FOLDER}/Datasets/yhat_t_power_dB.txt', 10*np.log10(y_hat/10).T, fmt='%.5f', delimiter=', ')
    def saveAll(self, model, cmap='magma'):
        self.saveSimDataset()
        self.saveSimData(model)
        self.plotAll(cmap=cmap)
        self.saveSelf() # Only useful if wanting a .mat file
    def plotBackprop(self, backprop_legend_location=0):
        plt.clf()
        labels_size = 20
        legend_size = 14
        tick_size = 12
        # Plot loss and accuracy values throughout training
        fig, ax = plt.subplots(figsize=(11.69, 8.27), dpi=100)
        ax.tick_params(axis='both', which='major', labelsize=tick_size)
        ax.tick_params(axis='both', which='minor', labelsize=tick_size)
        color = 'tab:blue'
        ax.set_xlabel('Epochs', fontsize=labels_size)
        ax.set_ylabel('$\mathcal{L}$', color=color, fontsize=labels_size)
        lns1 = ax.plot(self.losses, color='tab:blue', label='Losses')
        ax.tick_params(axis='y', labelcolor=color)
        ax2 = ax.twinx()  # instantiate a second axes that shares the same x-axis
        ax2.tick_params(axis='both', which='major', labelsize=tick_size) #Fixes tick size issue
        ax2.tick_params(axis='both', which='minor', labelsize=tick_size)
        color = 'tab:red'
        ax2.set_ylabel('Accuracy (\%)', color=color, fontsize=labels_size)  # we already handled the x-label with ax
        ax2.set_ylim([0, 100])
        lns2 = ax2.plot(self.val_accuracy, color='tab:red', label='Validation Accuracy')
        lns3 = ax2.plot(self.trn_accuracy, color='k', label='Training Accuracy')
        lns = lns1+lns2+lns3
        labs = [l.get_label() for l in lns]
        ax2.legend(lns, labs, loc=backprop_legend_location, fontsize=legend_size)
        ax2.tick_params(axis='y', labelcolor=color)
        fig.tight_layout() 
        plt.savefig(f'{self.FOLDER}/backprop_{self.topo}.pdf', bbox_inches="tight")
        plt.clf()
    def plotAll(self, cmap='gist_heat', trainingLoss=0.00):
        labels_size = 30
        legend_size = 30
        tick_size = 28
        contour_color = (0.36, 0.54, 0.66)
        contour_color2 = 'black'
        contour_linewidth = 3.5
        tick_fmt = '%.2f'
        # plt.rcParams['font.family'] = 'STIXGeneral'
        # rc('font', weight='bold',**{'family':'serif','serif':['Times New Roman']})
        # rc('text', usetex=True)
        # the above settings has no effect... has to use preamble to change fonts
        rc('text.latex', preamble=r'\usepackage{mathptmx}')

        # Plot Loss + Phase uncert accuracies
        # plt.pcolor(self.loss_dB, self.phase_uncert_theta, self.accuracy_LPU, vmin=100/(self.N+1)*0, vmax=100, cmap=cmap, rasterized=True)
        plt.figure(figsize=(6.95, 5.03)) # compress the graph (around) quarter in size, by cutting top half and compress horizontally
        plt.pcolor(self.loss_dB, self.phase_uncert_theta[:len(self.phase_uncert_theta)//2], self.accuracy_LPU[:self.accuracy_LPU.shape[0]//2,:], vmin=100/(self.N+1)*0, vmax=100, cmap=cmap, rasterized=True)

        ax = plt.gca()
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        ax.tick_params(axis='both', which='minor', labelsize=tick_size)
        ax.tick_params(axis='both', which='major', labelsize=tick_size)

        plt.xlabel('Loss/MZI (dB)', fontsize=labels_size)
        plt.ylabel(r'$\sigma_{\theta}, \sigma_\phi$ (Rad)', fontsize=labels_size)
        cbar = plt.colorbar()
        cbar.ax.tick_params(labelsize=tick_size)
        cbar.set_label('Accuracy (\%)', fontsize=labels_size)
        # plt.title(f'{self.N}$\\times${self.N} {self.topology}', fontsize=labels_size)
        plt.tight_layout()
        plt.savefig(f'{self.FOLDER}/Plots/LPU_ACC_{self.topo}_N={self.N}.pdf')
        plt.clf()

        # Plot Loss + Phase uncert accuracies along with contour of high accuracy region
        # plt.pcolor(self.loss_dB, self.phase_uncert_theta, self.accuracy_LPU, vmin=100/(self.N+1)*0, vmax=100, cmap=cmap, rasterized=True)
        plt.pcolor(self.loss_dB, self.phase_uncert_theta[:len(self.phase_uncert_theta)//2], self.accuracy_LPU[:self.accuracy_LPU.shape[0]//2,:], vmin=100/(self.N+1)*0, vmax=100, cmap=cmap, rasterized=True)

        ax = plt.gca()
        ax.tick_params(axis='both', which='minor', labelsize=tick_size)
        ax.tick_params(axis='both', which='major', labelsize=tick_size)
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))

        cbar = plt.colorbar()
        cbar.ax.tick_params(labelsize=tick_size)
        # cs = plt.contour(self.loss_dB, self.phase_uncert_theta, self.accuracy_LPU, [self.zeta*100], colors=[contour_color2], linewidths=contour_linewidth)
        cs = plt.contour(self.loss_dB, self.phase_uncert_theta[:len(self.phase_uncert_theta)//2], self.accuracy_LPU[:self.accuracy_LPU.shape[0]//2,:], [self.zeta*100], colors=[contour_color2], linewidths=contour_linewidth)
        proxy = [plt.Rectangle((0,0),1,1,fc = contour_color2)]
        # plt.legend(proxy, [f'Above {int(self.zeta*100)}\% Accuracy'], prop={'size': legend_size})
        plt.annotate(f'FoM = 0.XXX dB·rad', xy=(0.099, 0.453), size=legend_size, bbox=dict(boxstyle='square,pad=0.1', alpha=0.75, facecolor='w'))
                    #  (get FoM with script)    0.288 0.420 for fontsize of 50
        plt.text(-0.28, -0.12, f'h)', size=legend_size) # change numbering of plot
        plt.xlabel('Loss/MZI (dB)', fontsize=labels_size)
        plt.ylabel(r'$\sigma_{\theta}, \sigma_\phi$ (Rad)', fontsize=labels_size)
        cbar.set_label('Accuracy (\%)', fontsize=labels_size)
        plt.title(f'Bokun', fontsize=labels_size) # change title of plot (topology name)
        # plt.title(f'{self.N}$\\times${self.N} {self.topology}', fontsize=labels_size)
        plt.tight_layout()
        plt.savefig(f'{self.FOLDER}/Plots/LPU_ACC_Contour_{self.topo}_N={self.N}.pdf')
        plt.clf()
        df = pd.DataFrame(columns=["Theta/Phi", "Loss", "Accuracy"])
        temp_x = 0
        temp_y = 0
        temp = 0
        for x in self.phase_uncert_theta:
            for y in self.loss_dB:
                df.loc[temp] = [x, y, self.accuracy_LPU[temp_x][temp_y]]
                temp_y = temp_y + 1
                temp = temp + 1
            temp_x = temp_x + 1
            temp_y = 0
        # print(df.to_string())
        df.to_csv(f'{self.FOLDER}/{self.topo}_LPU_results.csv')

        plt.figure(figsize=(6.69, 4.90)) # cut the graph (around) quarter in size
        # Plot Phase uncert accuracies along with contour of high accuracy region
        # plt.pcolor(self.phase_uncert_theta, self.phase_uncert_phi, self.accuracy_PT, vmin=100/(self.N+1)*0, vmax=100, cmap=cmap, rasterized=True)
        plt.pcolor(self.phase_uncert_theta[:len(self.phase_uncert_theta)//2], self.phase_uncert_phi[:len(self.phase_uncert_phi)//2], self.accuracy_PT[:self.accuracy_PT.shape[0]//2,:self.accuracy_PT.shape[1]//2], vmin=100/(self.N+1)*0, vmax=100, cmap=cmap, rasterized=True)

        ax = plt.gca()
        ax.tick_params(axis='both', which='minor', labelsize=tick_size)
        ax.tick_params(axis='both', which='major', labelsize=tick_size)
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))

        cbar = plt.colorbar()
        cbar.ax.tick_params(labelsize=tick_size)
        # cs = plt.contour(self.phase_uncert_theta, self.phase_uncert_theta, self.accuracy_PT, [self.zeta*100], colors=[contour_color2], linewidths=contour_linewidth)
        cs = plt.contour(self.phase_uncert_theta[:len(self.phase_uncert_theta)//2], self.phase_uncert_theta[:len(self.phase_uncert_theta)//2], self.accuracy_PT[:self.accuracy_PT.shape[0]//2,:self.accuracy_PT.shape[1]//2], [self.zeta*100], colors=[contour_color2], linewidths=contour_linewidth)
        proxy = [plt.Rectangle((0,0),1,1,fc = contour_color2)]
        # plt.legend(proxy, [f'Above {int(self.zeta*100)}\% Accuracy'], prop={'size': legend_size})
        plt.annotate(f'FoM = 0.XXX rad\u00b2', xy=(0.087, 0.451), size=legend_size, bbox=dict(boxstyle='square,pad=0.1', alpha=0.75, facecolor='w'))
                    #  (get FoM with script)
        plt.text(-0.14, -0.13, f'd)', size=legend_size) # change numbering
        plt.xlabel(r'$\sigma_\theta$ (Rad)', fontsize=labels_size)
        plt.ylabel(r'$\sigma_{\phi}$ (Rad)', fontsize=labels_size)
        cbar.set_label('Accuracy (\%)', fontsize=labels_size)
        plt.title(f'Bokun', fontsize=labels_size) # change title of plot (topology name)
        # plt.title(f'{self.N}$\\times${self.N} {self.topology}', fontsize=labels_size)
        plt.tight_layout()
        plt.savefig(f'{self.FOLDER}/Plots/PT_ACC_Contour_{self.topo}_N={self.N}.pdf')
        plt.clf()

        # Colormap of Phi + Theta phase uncertainty
        plt.pcolor(self.phase_uncert_theta[:len(self.phase_uncert_theta)//2], self.phase_uncert_phi[:len(self.phase_uncert_phi)//2], self.accuracy_PT[:self.accuracy_PT.shape[0]//2,:self.accuracy_PT.shape[1]//2], vmin=100/(self.N+1)*0, vmax=100, cmap=cmap, rasterized=True)
        # plt.pcolor(self.phase_uncert_theta, self.phase_uncert_phi, self.accuracy_PT, vmin=100/(self.N+1)*0, vmax=100, cmap=cmap, rasterized=True)

        ax = plt.gca()
        ax.tick_params(axis='both', which='minor', labelsize=tick_size)
        ax.tick_params(axis='both', which='major', labelsize=tick_size)
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))

        plt.xlabel(r'$\sigma_\theta$ (Rad)', fontsize=labels_size)
        plt.ylabel(r'$\sigma_{\phi}$ (Rad)', fontsize=labels_size)
        cbar = plt.colorbar()
        cbar.ax.tick_params(labelsize=tick_size)
        cbar.set_label('Accuracy (\%)', fontsize=labels_size)
        # plt.title(f'{self.N}$\\times${self.N} {self.topology}', fontsize=labels_size)
        plt.tight_layout()
        plt.savefig(f'{self.FOLDER}/Plots/PT_ACC_{self.topo}_N={self.N}.pdf')
        plt.clf()
        df = pd.DataFrame(columns=["Theta", "Phi", "Accuracy"])
        temp_x = 0
        temp_y = 0
        temp = 0
        for x in self.phase_uncert_theta:
            for y in self.phase_uncert_phi:
                df.loc[temp] = [x, y, self.accuracy_PT[temp_x][temp_y]]
                temp_y = temp_y + 1
                temp = temp + 1
            temp_x = temp_x + 1
            temp_y = 0
        # print(df.to_string())
        df.to_csv(f'{self.FOLDER}/{self.topo}_results.csv')
    def createFOLDER(self):
        if not os.path.isdir(self.FOLDER):
            os.makedirs(self.FOLDER)
        if not os.path.isdir(self.FOLDER + '/Datasets'):
            os.makedirs(self.FOLDER + '/Datasets')
        if not os.path.isdir(self.FOLDER + '/Topologies'):
            os.makedirs(self.FOLDER + '/Topologies')
        if not os.path.isdir(self.FOLDER + '/Plots'):
            os.makedirs(self.FOLDER + '/Plots')
    def pickle_save(self):
        with open(f'{self.FOLDER}/{self.topo}.pkl', 'wb') as p:
            pickle.dump(self, p)
    def pickle_load(self, onn_folder = None):
        if onn_folder is not None:
            self.FOLDER = onn_folder
        with open(f'{self.FOLDER}/{self.topo}.pkl', 'rb') as p:
            self = pickle.load(p)
            return self 
