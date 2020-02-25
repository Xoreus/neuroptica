"""
Saves dataset to FOLDER

Author: Simon Geoffroy-Gagnon
Edit: 15.01.15
"""
import os

def createFOLDER(FOLDER):
    if not os.path.isdir(FOLDER):
        os.makedirs(FOLDER)
    if not os.path.isdir(FOLDER + '/Figures_Fitting'):
        os.makedirs(FOLDER + '/Figures_Fitting')
    if not os.path.isdir(FOLDER + '/Data_Fitting'):
        os.makedirs(FOLDER + '/Data_Fitting')
    if not os.path.isdir(FOLDER + '/Phases'):
        os.makedirs(FOLDER + '/Phases')
    if not os.path.isdir(FOLDER + '/Datasets'):
        os.makedirs(FOLDER + '/Datasets')
    if not os.path.isdir(FOLDER + '/Losses_per_MZI'):
        os.makedirs(FOLDER + '/Losses_per_MZI')
    if not os.path.isdir(FOLDER + '/TransformationMatrices'):
        os.makedirs(FOLDER + '/TransformationMatrices')
    if not os.path.isdir(FOLDER + '/Topologies'):
        os.makedirs(FOLDER + '/Topologies')
