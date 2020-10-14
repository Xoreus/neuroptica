import ONN_Simulation_Class as ONN_Cls
import sys
sys.path.append('../')
import neuroptica as neu

# Initialize onn class
onn = ONN_Cls.ONN_Simulation() # Required for containing training/simulation information
# Define what folder we are loading
onn.FOLDER = f'/home/simon/Documents/neuroptica/tests/Analysis/3l_cReLU_10x10/N=10_20dBRange'
# Define what topology we are loading 
onn.topo = 'E_P'
# Load folder
onn = onn.pickle_load()

print(onn.accuracy_LPU)
print(onn.accuracy_PT)
# Replot everything
onn.plotAll(backprop_legend_location=1) # plot training and tests

''' Backprop Legend Location Codes:
'best' 	        0
'upper right' 	1
'upper left' 	2
'lower left' 	3
'lower right' 	4
'right'  	5
'center left' 	6
'center right' 	7
'lower center' 	8
'upper center' 	9
'center'    	10
'''

