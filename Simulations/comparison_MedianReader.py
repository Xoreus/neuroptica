''' phase_uncert_thetar simulating Optical Neural Network

using Neuroptica and linearly separable datasets
Now goes over every topology types with N = 4-32

Author: Simon Geoffroy-Gagnon
Edit: 2020.03.28
'''
import numpy as np
from sklearn.preprocessing import MinMaxScaler as mms
import calculate_accuracy as calc_acc
import ONN_Simulation_Class as ONN_Cls
import training_onn as train
import test_trained_onns as test
import ONN_Setups
import create_datasets
from copy import deepcopy
from plot_scatter_matrix import plot_scatter_matrix
import matplotlib.pyplot as plt
import neuroptica as neu
import pandas as pd

FOLDER = f'Analysis/2. For Paper/N=10 MEDIAN 2 CCE MNIST DOUBLE 100 TIMES/'
loss_dB_Values = np.linspace(0, 5, 126)
onn_topo = ['Diamond', 'Clements', 'Reck']

Diamond_Mesh = pd.read_csv(FOLDER + '/Diamond_results.csv')
Clements_Mesh = pd.read_csv(FOLDER + '/Clements_results.csv')
Reck_Mesh = pd.read_csv(FOLDER + '/Reck_results.csv')

Diamond_Mesh.rename(columns={'Unnamed: 0':''}, inplace=True)
Diamond_Mesh.set_index('', inplace=True)

Clements_Mesh.rename(columns={'Unnamed: 0':''}, inplace=True)
Clements_Mesh.set_index('', inplace=True)

Reck_Mesh.rename(columns={'Unnamed: 0':''}, inplace=True)
Reck_Mesh.set_index('', inplace=True)

Diamond_Mesh_Values_Med = []
Diamond_Mesh_Values_Max = []
Diamond_Mesh_Values_Min = []

Clements_Mesh_Values_Med = []
Clements_Mesh_Values_Max = []
Clements_Mesh_Values_Min = []

Reck_Mesh_Values_Med = []
Reck_Mesh_Values_Max = []
Reck_Mesh_Values_Min = []


for i in loss_dB_Values:
    Diamond_Mesh_Values_Med.append(Diamond_Mesh.loc['Median'][str(i)]) 
    Diamond_Mesh_Values_Max.append(Diamond_Mesh.loc['Max'][str(i)]) 
    Diamond_Mesh_Values_Min.append(Diamond_Mesh.loc['Min'][str(i)]) 
    Clements_Mesh_Values_Med.append(Clements_Mesh.loc['Median'][str(i)]) 
    Clements_Mesh_Values_Max.append(Clements_Mesh.loc['Max'][str(i)]) 
    Clements_Mesh_Values_Min.append(Clements_Mesh.loc['Min'][str(i)]) 
    Reck_Mesh_Values_Med.append(Reck_Mesh.loc['Median'][str(i)]) 
    Reck_Mesh_Values_Max.append(Reck_Mesh.loc['Max'][str(i)]) 
    Reck_Mesh_Values_Min.append(Reck_Mesh.loc['Min'][str(i)]) 

labels_size = 20
legend_size = 16
tick_size = 14
color = 'tab:blue'
fig, ax = plt.subplots(figsize=(8.27, 8.27), dpi=100) #11.69, 8.27
ax.tick_params(axis='both', which='major', labelsize=tick_size)
ax.tick_params(axis='both', which='minor', labelsize=tick_size)
ax.set_title("Median Validation Accuracy vs Loss/MZI (dB)", fontsize=labels_size)
ax.set_xlabel('Loss/MZI (dB)', fontsize=labels_size)
ax.set_ylabel("Validation Accuracy (\%)", fontsize=labels_size)
lns0 = ax.plot(loss_dB_Values, Diamond_Mesh_Values_Med, color='#edb120', label=onn_topo[0])
lns1 = ax.plot(loss_dB_Values, Clements_Mesh_Values_Med, color='#d95319', label=onn_topo[1])
lns2 = ax.plot(loss_dB_Values, Reck_Mesh_Values_Med, color='#0072bd', label=onn_topo[2])
ax.set_ylim([0, 100])
lns = lns0+lns1+lns2
labs = [l.get_label() for l in lns]
ax.legend(lns, labs, loc=0, fontsize=legend_size)
fig.tight_layout() 
plt.savefig(FOLDER + '/comparison_Median.pdf')
plt.clf()

labels_size = 20
legend_size = 16
tick_size = 14
color = 'tab:blue'
fig, ax = plt.subplots(figsize=(8.27, 8.27), dpi=100) #11.69, 8.27
ax.tick_params(axis='both', which='major', labelsize=tick_size)
ax.tick_params(axis='both', which='minor', labelsize=tick_size)
ax.set_title("Max Validation Accuracy vs Loss/MZI (dB)", fontsize=labels_size)
ax.set_xlabel('Loss/MZI (dB)', fontsize=labels_size)
ax.set_ylabel("Validation Accuracy (\%)", fontsize=labels_size)
lns0 = ax.plot(loss_dB_Values, Diamond_Mesh_Values_Max, color='#edb120', label=onn_topo[0])
lns1 = ax.plot(loss_dB_Values, Clements_Mesh_Values_Max, color='#d95319', label=onn_topo[1])
lns2 = ax.plot(loss_dB_Values, Reck_Mesh_Values_Max, color='#0072bd', label=onn_topo[2])
ax.set_ylim([0, 100])
lns = lns0+lns1+lns2
labs = [l.get_label() for l in lns]
ax.legend(lns, labs, loc=0, fontsize=legend_size)
fig.tight_layout() 
plt.savefig(FOLDER + '/comparison_Max.pdf')
plt.clf()

labels_size = 20
legend_size = 16
tick_size = 14
color = 'tab:blue'
fig, ax = plt.subplots(figsize=(8.27, 8.27), dpi=100) #11.69, 8.27
ax.tick_params(axis='both', which='major', labelsize=tick_size)
ax.tick_params(axis='both', which='minor', labelsize=tick_size)
ax.set_title("Min Validation Accuracy vs Loss/MZI (dB)", fontsize=labels_size)
ax.set_xlabel('Loss/MZI (dB)', fontsize=labels_size)
ax.set_ylabel("Validation Accuracy (\%)", fontsize=labels_size)
lns0 = ax.plot(loss_dB_Values, Diamond_Mesh_Values_Min, color='#edb120', label=onn_topo[0])
lns1 = ax.plot(loss_dB_Values, Clements_Mesh_Values_Min, color='#d95319', label=onn_topo[1])
lns2 = ax.plot(loss_dB_Values, Reck_Mesh_Values_Min, color='#0072bd', label=onn_topo[2])
ax.set_ylim([0, 100])
lns = lns0+lns1+lns2
labs = [l.get_label() for l in lns]
ax.legend(lns, labs, loc=0, fontsize=legend_size)
fig.tight_layout() 
plt.savefig(FOLDER + '/comparison_Min.pdf')
plt.clf()


