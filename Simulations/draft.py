#%% MZI, MZILayer, and MZIMesh classes
import numpy as np
import ONN_Simulation_Class as ONN_Cls
import sys
sys.path.append('../')
import neuroptica as neu
from main import save_onn
import matplotlib.ticker as tck
import matplotlib.pyplot as plt
from scipy.linalg import qr
plt.rcParams.update(plt.rcParamsDefault)
# matplotlib.rcParams['font.family'] = "sans-serif"
np.set_printoptions(precision=3, suppress=True, linewidth=np.inf)
np.random.seed(0)
'''
Adapted from Neuroptica, models the linear weight matrix expressed by an MZI mesh
contains 3 classess: "MZI", "MZILayer", and "MZIMesh"
Input:
- a set of MZIs and their phase shifter states
- the number of waveguides in the mesh
- the location of each MZI in the mesh
Output:
- The linear transformation matrix of this mesh.

Author: bokunzhao
'''

class MZI: # individual MZI
    def __init__(self, rho_1=0.5, rho_2=0.5, theta=None, phi=None, m=0, n=0) -> None:
        self.rho_1 = rho_1 # first coupler splitting ratio
        self.rho_2 = rho_2 # second coupler splitting ratio (applies to top branch only?)
        self.theta = theta # interal phase shifter value, unit: rad
        self.phi = phi # external phase shifter value, unit: rad
        self.m = m # aligned port number of the first branch
        self.n = n # aligned port number of the second branch
        if theta == None:
            self.theta = 2*np.pi*np.random.rand()
        if phi == None:
            self.phi = 2*np.pi*np.random.rand()
        # first coupler (splitter)
        self.coupler_1 = np.array([[np.sqrt(rho_1), 1j*np.sqrt(1-rho_1)],
                                   [1j*np.sqrt(1-rho_1), np.sqrt(rho_1)]])
        # phase shifter θ
        self.shifter_1 = np.array([[np.exp(1j*self.theta), 0],
                                   [0, 1]])
        # second coupler (combiner)
        self.coupler_2 = np.array([[np.sqrt(rho_2), 1j*np.sqrt(1-rho_2)],
                                   [1j*np.sqrt(1-rho_2), np.sqrt(rho_2)]])
        # phase shifter φ
        self.shifter_2 = np.array([[np.exp(1j*self.phi), 0],
                                   [0, 1]])
        # transfer matrix: ground up from individual components
        self.D_mzi = self.shifter_2@self.coupler_2@self.shifter_1@self.coupler_1
        # transfer matrix: from BookChapter_Final2.pdf, Eq. 1.1, currently used in Neuroptica. Verified to be the same as the above line
        # self.D_mzi = 0.5 * np.array([[np.exp(1j * self.phi) * (np.exp(1j * self.theta) - 1),
        #     1j * np.exp(1j * self.phi) * (1 + np.exp(1j * self.theta))],
        #     [1j * (np.exp(1j * self.theta) + 1), 1 - np.exp(1j * self.theta)]
        #     ], dtype=np.complex128)
        # transfer matrix: reversed MZI with phi phase shifter on the input
        # self.D_mzi = self.coupler_2@self.shifter_1@self.coupler_1@self.shifter_2
    def set_phase_shift(self, new_theta=None, new_phi=None, new_rho_1=0.5, new_rho_2=0.5):
        self.theta = new_theta
        self.phi = new_phi
        if new_theta == None:
            self.theta = 2*np.pi*np.random.rand()
        if new_phi == None:
            self.phi = 2*np.pi*np.random.rand()
        self.rho_1 = new_rho_1
        self.rho_2 = new_rho_2
        self.coupler_1 = np.array([[np.sqrt(self.rho_1), 1j*np.sqrt(1-self.rho_1)],
                                   [1j*np.sqrt(1-self.rho_1), np.sqrt(self.rho_1)]])
        self.shifter_1 = np.array([[np.exp(1j*self.theta), 0],
                                   [0, 1]])
        self.coupler_2 = np.array([[np.sqrt(self.rho_2), 1j*np.sqrt(1-self.rho_2)],
                                   [1j*np.sqrt(1-self.rho_2), np.sqrt(self.rho_2)]])
        self.shifter_2 = np.array([[np.exp(1j*self.phi), 0],
                                   [0, 1]])
        self.D_mzi = self.shifter_2@self.coupler_2@self.shifter_1@self.coupler_1
    def get_D_mzi(self):
        return self.D_mzi
    def get_DH_mzi(self, hubert_dim, top_port_alignment):
        # D_mzi placed in a larger Hubert matrix (Book Ch2 Apporach)
        # VERIFIED: same as columnwise T_matrix approach
        DH_mzi = np.eye(hubert_dim, dtype=np.complex64) # NxN identity matrix
        DH_mzi[top_port_alignment:top_port_alignment+2,
                      top_port_alignment:top_port_alignment+2] = self.D_mzi
        self.DH_mzi = DH_mzi
        return self.DH_mzi
      
class MZILayer: # "column" of MZIs
    def __init__(self, MZI_list:list[MZI], num_waveguides:int, aligned_port_numbers:list[int]) -> None:
        '''Example:
        # num_waveguides = 10   ([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        # aligned_port_numbers = [   1,    3,    5,    7      ]
        # MZI_list =              [MZI1, MZI2, MZI3, MZI4]
        # (This corresponds to the second MZI column in a 10x10 Clements Mesh)
        '''
        assert len(MZI_list) == len(aligned_port_numbers), f"No. of MZIs({len(MZI_list)}) does not match no. of port numbers({len(aligned_port_numbers)})!"
        assert max(aligned_port_numbers) < num_waveguides-1, "max aligned port number should be one less than max available port number"
        self.width = num_waveguides # i.e. number of waveguides
        self.MZIs = MZI_list
        self.aligned_port_numbers = aligned_port_numbers
    def get_transfer_matrix(self):
        hubert_space = np.eye(self.width, dtype=np.complex64) # NxN identity matrix
        for i, eachMZI in enumerate(self.MZIs):
            hubert_space[self.aligned_port_numbers[i]:self.aligned_port_numbers[i]+2,
                              self.aligned_port_numbers[i]:self.aligned_port_numbers[i]+2] = eachMZI.get_D_mzi()
        self.T_matrix = hubert_space
        return self.T_matrix

class MZIMesh: # a mesh of MZIs made up of MZI columns
    def __init__(self, _profile:list[list[int]], _phases:np.ndarray=None, _no_wg:int=10) -> None:
        mzi_list = []
        mzi_column = []
        mzi_num = 0 # keep track of total number of mzis constructed (in this mesh)
        for k, column in enumerate(_profile): # construct each mzi column
            for mzi_count in column[::2]: # construct each mzi in that column
                if _phases == None: # random initial phase
                    mzi_list.append(MZI(m=mzi_count, n=mzi_count+1))
                else:
                    mzi_list.append(MZI(theta=_phases[mzi_num][0], phi=_phases[mzi_num][1], m=mzi_count, n=mzi_count+1))
                mzi_num += 1 # constructed mzi +1
            mzi_column.append(MZILayer(mzi_list, _no_wg, column[::2]))
            mzi_list = []
        for i, each_col in enumerate(mzi_column):
            assert each_col.width == _no_wg, f"MZI layer {i} has different number of waveguides."
        self.MZI_Layers = mzi_column
        self.no_wg = _no_wg
        self.num_mzis = mzi_num
    def get_weight_matrix(self):
        transformation_matrix = np.eye(self.no_wg, dtype=np.complex64)
        for j, each in enumerate(self.MZI_Layers[::-1]):
            # print(each.get_transfer_matrix())
            transformation_matrix = transformation_matrix @ each.get_transfer_matrix()
        self.W_matrix = transformation_matrix
        return self.W_matrix
    def get_weight_matrix_with_DH(self):
        # get W from DH_mzis directly, not from T_matrices
        # VERIFIED: same as columnwise T_matrix approach
        transformation_matrix = np.eye(self.no_wg, dtype=np.complex64)
        MZI_tuples = [] # list of (<MZILayer>, port_num_index, <MZI>, colNum)
        for colNum, eachCol in enumerate(self.MZI_Layers[::-1]):
            for port_num_idx, eachMZI in enumerate(eachCol.MZIs):
                # transformation_matrix = transformation_matrix @ eachMZI.get_DH_mzi(self.no_wg, eachCol.aligned_port_numbers[port_num_idx])
                MZI_tuples.append((eachCol, port_num_idx, eachMZI, len(self.MZI_Layers) - colNum))
        # MZI_tuples[2],MZI_tuples[3]=MZI_tuples[3],MZI_tuples[2]
        for each in MZI_tuples:
            print(f"Column {each[3]}, aligned port {each[0].aligned_port_numbers[each[1]]}, ({each[2].theta:.3f}, {each[2].phi:.3f})")
            transformation_matrix = transformation_matrix @ each[2].get_DH_mzi(self.no_wg, each[0].aligned_port_numbers[each[1]])
        self.W_matrix_DH = transformation_matrix
        return self.W_matrix_DH
    def set_phases(self, new_phases=None):
        if new_phases == None: # set to random
            i = 0
            for each_MZIColumn in self.MZI_Layers:
                for each_MZI in each_MZIColumn.MZIs:
                    each_MZI.set_phase_shift()
                    i+=1
        else: # set with provided
            assert len(new_phases) == self.num_mzis, f"Number of provided phase pair ({len(new_phases)}) is different from number of MZIs in the mesh ({self.num_mzis})."
            i = 0
            for each_MZIColumn in self.MZI_Layers:
                for each_MZI in each_MZIColumn.MZIs:
                    each_MZI.set_phase_shift(new_phases[i][0],new_phases[i][1])
                    i+=1

def see_each_mzi(mesh:MZIMesh):
    '''
    Helper Function to visualize each MZI's (sigma_theta, sigma_phi, loss)

    sigma is the standard deviation of the normal distribution from which the phase error is drawn
    see the method "get_transfer_matrix()" in components.py
    '''
    print("\n------------------------------------------------------------------------------------------------")
    # print(f"There are {1} MZImesh(es) in the model.")
    min_theta = np.inf
    max_theta = -np.inf
    min_phi = np.inf
    max_phi = -np.inf
    mesh_count=1
    for i in range(mesh_count):
        mzi_layers = mesh.MZI_Layers # list of objects <MZILayer>
        print(f"In MZImesh {i}, there are {len(mzi_layers)} MZILayers (columns)")
        for col_idx, eachMZILayer in enumerate(mzi_layers):
            for row_idx, eachMZI in enumerate(eachMZILayer.MZIs):
                # you can also print other information related to each MZI here, such as the theta/phi phases...
                mzi_info = f" [{eachMZI.m}____{eachMZI.n}] "
                # mzi_info = f" [{eachMZI.theta}____{eachMZI.phi}] "
                stagger = " "*(len(mzi_info)//2) # to better visualize the zig-zag alignment between each MZI column
                if row_idx==0: # at the start of each MZI column
                    print(stagger*eachMZI.m,end="") # amount of visual offset to print
                else:
                    print(stagger*(eachMZI.m-previousMZI_n-1),end="")
                print(mzi_info, end="")
                if eachMZI.theta < min_theta: min_theta = eachMZI.theta
                if eachMZI.phi < min_phi: min_phi = eachMZI.phi
                if eachMZI.theta > max_theta: max_theta = eachMZI.theta
                if eachMZI.phi > max_phi: max_phi = eachMZI.phi
                previousMZI_n = eachMZI.n
            print("\n")
    print(f"theta range: ({min_theta},{max_theta})")
    print(f"phi range: ({min_phi},{max_phi})")
    print("------------------------------------------------------------------------------------------------")

def get_F1_score(cls_, gt_):
    # Calculate True Positives, False Positives, and False Negatives
    TP = np.sum((cls_ == 1) & (gt_ == 1))
    FP = np.sum((cls_ == 1) & (gt_ == 0))
    TN = np.sum((cls_ == 0) & (gt_ == 0))
    FN = np.sum((cls_ == 0) & (gt_ == 1))
    # Precision and Recall
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    # F1 Score
    if (precision + recall) > 0:
        f1 = 2 * (precision * recall) / (precision + recall)
    else:
        f1 = 0
    # P_trigger = (FP + TP)/(TN + FP + FN + TP)
    return f1

def get_statistics(y):
    # y is a the result_array (#mesh, #tst_inst, #ports)
    minimum = np.min(y, axis=0) # expected shape (8)
    q1 = np.quantile(y, 0.25, axis=0)
    median = np.median(y, axis=0)
    q3 = np.quantile(y, 0.75, axis=0)
    maximum = np.max(y, axis=0)
    mean = np.quantile(y, 0.50, axis=0)
    return np.vstack((minimum, q1, median, q3, maximum, mean)) # shape (6, 8)

def generate_random_unitary(dim):
    """
    Generates a random unitary matrix of a given dimension.
    Args:
        dim (int): The dimension of the square unitary matrix.
    Returns:
        numpy.ndarray: A complex numpy array representing a random unitary matrix.
    """
    if not isinstance(dim, int) or dim < 1:
        raise ValueError("Dimension must be a positive integer.")

    # Generate a complex Gaussian random matrix
    # Real and imaginary parts are drawn from a standard normal distribution
    random_matrix = np.random.randn(dim, dim) + 1j * np.random.randn(dim, dim)

    # Perform QR decomposition
    Q, R = qr(random_matrix)

    # The diagonal elements of R can have arbitrary phases.
    # To ensure Q is truly unitary (and for a more uniform distribution),
    # we adjust the phases of Q based on the diagonal of R.
    # This effectively normalizes the columns of the original random_matrix
    # to unit length while preserving orthogonality.
    diag_R = np.diag(R)
    phases = diag_R / np.abs(diag_R)
    unitary_matrix = Q @ np.diag(phases)

    return unitary_matrix

# Example usage:
# dimension = 3
# random_unitary = generate_random_unitary(dimension)
# print("Random Unitary Matrix:")
# print(random_unitary)

# Verify if it's unitary (U @ U_dagger should be close to identity)
# identity_check = random_unitary @ random_unitary.conj().T
# print("\nIdentity Check (U @ U_dagger):")
# print(identity_check)
#%% sample topology profiles
clements_8x8_profile  = [[0, 1, 2, 3, 4, 5, 6, 7],
                        [   1, 2, 3, 4, 5, 6  ],
                        [0, 1, 2, 3, 4, 5, 6, 7],
                        [   1, 2, 3, 4, 5, 6  ],
                        [0, 1, 2, 3, 4, 5, 6, 7],
                        [   1, 2, 3, 4, 5, 6  ],
                        [0, 1, 2, 3, 4, 5, 6, 7],
                        [   1, 2, 3, 4, 5, 6  ]]
miniBokun_8x8_profile= [[    2,3,4,5    ],
                        [  1,2,3,4,5,6  ],
                        [0,1,2,3,4,5,6,7],
                        [  1,2,3,4,5,6  ],
                        [    2,3,4,5    ]]
triangle_8x8_profile=  [[0,1,2,3,4,5,6,7],
                        [  1,2,3,4,5,6  ],
                        [    2,3,4,5    ],
                        [      3,4      ]]

miniBokun_10x10_profile =[[    2,3,4,5,6,7    ],
                          [  1,2,3,4,5,6,7,8  ],
                          [0,1,2,3,4,5,6,7,8,9],
                          [  1,2,3,4,5,6,7,8  ],
                          [    2,3,4,5,6,7    ],
                          [      3,4,5,6      ]]
triangle_6x6_profile = [[  0,1,2,3,4,5  ],
                        [    1,2,3,4    ],
                        [      2,3      ]]
clements_4x4_profile  = [[0, 1, 2, 3],
                        [   1, 2,  ],
                        [0, 1, 2, 3],
                        [   1, 2,  ]]
reck_4x4_profile     = [[0, 1      ],
                        [   1, 2,  ],
                        [0, 1, 2, 3],
                        [   1, 2   ],
                        [0, 1      ]]

clements_6x6_profile  = [[0, 1, 2, 3, 4, 5],
                        [    1, 2, 3, 4 ],
                        [ 0, 1, 2, 3, 4, 5],
                        [    1, 2, 3, 4 ],
                        [ 0, 1, 2, 3, 4, 5],
                        [    1, 2, 3, 4 ]]
clements_6x6_profile_m1  = [[0, 1, 2, 3, 4, 5],
                            [    1, 2, 3, 4 ],
                            [ 0, 1, 2, 3, 4, 5],
                            [    1, 2, 3, 4 ],
                            [ 0, 1, 2, 3, 4, 5],
                            [    1, 2, 3, 4 ]]

phases = [(4.8864483594755725, 1.492515503572874),
          (5.17909476544146, 6.067981171564244), 
          (6.111033028633725, 2.849105648924097), 
          (3.8267266534701316, 4.872776801893368), 
          (4.031375540680534, 5.73787266515462), 
          (0.2201409734487944, 1.8752133304264136)]
        #   (1.6968847452268525,4.536574331216701),
        #   (5.716320448475244,0.46175647746863463)]

a_clement_mesh = MZIMesh(clements_4x4_profile, _phases=phases, _no_wg=4)
see_each_mzi(a_clement_mesh)
print(f"clement weight:\n{a_clement_mesh.get_weight_matrix()}")
# weight_values = a_clement_mesh.get_weight_matrix().round(3).flatten()
# count = 0
# for each in weight_values:
#     if count % 8 == 7: # print 7 values per line
#         print(f"{each:.3f}\\\\")
#     else:
#         print(f"{each:.3f}&", end="")
#     count += 1
#     print(f"{each.real:.3f}+{each.imag:.3f}j", end=", ")
# print(f"clement weight:\n{a_clement_mesh.get_weight_matrix().flatten()}")
# a_miniBokun_mesh = MZIMesh(miniBokun_8x8_profile, _phases=None, _no_wg=8)
# print(f"miniBokun weight:\n{a_miniBokun_mesh.get_weight_matrix()}")

# import pandas as pd
# clement_weight_matrix = np.round(a_clement_mesh.get_weight_matrix(), decimals=3)
# df = pd.DataFrame(clement_weight_matrix)
# # Convert to DataFrame
# def format_complex(c):
#     r = int(c.real) if c.real.is_integer() else round(c.real, 5)
#     i = int(c.imag) if c.imag.is_integer() else round(c.imag, 5)
#     return f"{r}+{i}j" if i >= 0 else f"{r}{i}j"
#     # Format: remove trailing ".0" and format nicely

# # Apply formatting
# df_str = df.applymap(format_complex)
# print(f"clement weight:\n{df_str}")
# # Save to Excel (preserves 2D layout)
# df_str.to_csv('output.csv', index=False, header=False)


#%% Expressivity Study based on mass sampling
topo = 'miniBokun'
no_tst_meshes = 10000 # 10000
no_tst_inst = 10000
total_power = 10

'''======================test with uniform input========================'''
# test_random_unitary = True # if true, use untiary matrix with entries being randomized directly instead of ones from a MZI mesh (whose entries are randomized indirectly via the phase shifters)
# no_tst_inst = 1
# uniform_sample = np.ones((1, 8), dtype=np.float32)* (np.sqrt(10/8))
# print(uniform_sample)
# print(np.sum(uniform_sample**2)) # total power per sample
# # allocate array to store detected power at each port of each sample, of each mesh
# # (#mesh, #tst_inst, #ports)
# result_array = np.empty((no_tst_meshes,no_tst_inst,8),dtype=np.float32)

# my_onn = MZIMesh(triangle_8x8_profile, _phases=None, _no_wg=8)
# for i in range(no_tst_meshes):
#     # print(f"testing with {topo} mesh {i} port {port_choice}")
#     if test_random_unitary:
#         W = generate_random_unitary(8)
#         is_rand_unit = '_RANDOM_UNITARY'
#     else:
#         W = my_onn.get_weight_matrix() # The unitary weight matrix represented by the mesh
#     # yhat = np.matmul(W, port_concentrated_samples[:no_tst_inst, :].T)# add [8//2-1:8//2+1 ,:] if keep middle two ports
#     # yhat = np.abs(yhat) ** 2 # complex phasor -> real intensity (power)
#     yhat = np.abs(np.matmul(W, uniform_sample[:no_tst_inst, :].T)) ** 2 # one-liner
#     result_array[i] = yhat.T
#     # np.save(f'./Analysis/iris_augment/EXPRESSIVITY_STUDY/EXP_clements_output_{i}', yhat)
#     my_onn.set_phases() # shuffle unitary mesh weights
# # np.save(f'./Analysis/iris_augment/EXPRESSIVITY_STUDY/EXP_clements_output', result_array)
# stat_quantities = get_statistics(result_array.reshape(-1, 8))
# # print(stat_quantities)
# np.save(f'./Analysis/iris_augment/EXPRESSIVITY_STUDY/EXP_{topo}_port_{-1}_statistics{is_rand_unit}', stat_quantities)

'''======================test with skewd input========================'''
test_random_unitary = True # if true, use untiary matrix with entries being randomized directly instead of ones from a MZI mesh (whose entries are randomized indirectly via the phase shifters)
power_skew_thres = 0.77 # samples with > [thres * maximum port power] will be retained as skew samples
raw_samples = np.random.rand(no_tst_inst, 8)
retrived_test_samples = np.sqrt(total_power*raw_samples / np.sum(raw_samples, axis=-1)[:, np.newaxis])
print(np.sum(retrived_test_samples**2, axis=-1)) # total power per sample
# retain samples with strongly-skewed power distribution only
max_power = retrived_test_samples.max() # max power appearing on a channel
no_tst_inst = 10 # updated to no. of remaining samples per port
print(retrived_test_samples.max(),retrived_test_samples.min())
#%% 
for port_choice in [0, 3, 5, 7]: # [0, 3, 5, 7]
    port_concentrated_samples = retrived_test_samples[retrived_test_samples[:, port_choice]>max_power*power_skew_thres]
    port_concentrated_samples=port_concentrated_samples[:no_tst_inst, :]
    print(port_concentrated_samples.shape)
    # allocate array to store detected power at each port of each sample, of each mesh
    # (#mesh, #tst_inst, #ports)
    result_array = np.empty((no_tst_meshes,no_tst_inst,8),dtype=np.float32)

    my_onn = MZIMesh(miniBokun_8x8_profile, _phases=None, _no_wg=8)
    for i in range(no_tst_meshes):
        # print(f"testing with {topo} mesh {i} port {port_choice}")
        if test_random_unitary:
            W = generate_random_unitary(8)
            is_rand_unit = '_RANDOM_UNITARY'
        else:
            W = my_onn.get_weight_matrix() # The unitary weight matrix represented by the mesh
        # yhat = np.matmul(W, port_concentrated_samples[:no_tst_inst, :].T)# add [8//2-1:8//2+1 ,:] if keep middle two ports
        # yhat = np.abs(yhat) ** 2 # complex phasor -> real intensity (power)
        yhat = np.abs(np.matmul(W, port_concentrated_samples[:no_tst_inst, :].T)) ** 2 # one-liner
        result_array[i] = yhat.T
        # np.save(f'./Analysis/iris_augment/EXPRESSIVITY_STUDY/EXP_clements_output_{i}', yhat)
        my_onn.set_phases() # shuffle unitary mesh weights
    # np.save(f'./Analysis/iris_augment/EXPRESSIVITY_STUDY/EXP_clements_output', result_array)
    stat_quantities = get_statistics(result_array.reshape(-1, 8))
    # print(stat_quantities)
    np.save(f'./Analysis/iris_augment/EXPRESSIVITY_STUDY/EXP_{topo}_port_{port_choice}_statistics{is_rand_unit}', stat_quantities)

'''=================================================================='''
# print(f"Calibration-order Weight matrix:\n{my_onn.get_weight_matrix_with_DH()}")
# my_onn.MZI_Layers[2].MZIs[1].set_phase_shift(0, 0) # set the 2nd mzi in 3rd column
# print(f"Weight matrix after update:\n{my_onn.get_weight_matrix()}")
# my_onn.MZI_Layers[2].MZIs[1].set_phase_shift(1.34, 5.72) # revert it back
# print(f"Weight matrix after reverting:\n{my_onn.get_weight_matrix()}")

#%% Exploring individual MZI's routing capability
'''
Exploring individual MZI's routing capability, especially when given:
- non-coherent light between two input ports (phase diff ≠ 0)
- non 50:50 coupler
- see update 2024_05_01
'''
num_points = 1000
theta_sweep = np.linspace(0, 2*np.pi, num_points)
phi_sweep = np.linspace(1, 2*np.pi, 1)

A_top = np.sqrt(0.3) # [0, 1] for relative amplitude
A_bottom = np.sqrt(1-A_top**2)
top_input_phase = 1.20 # [0, 2π)
bottom_input_phase = 0.00 # [0, 2π)

E_in_top = A_top*np.exp(1j*(top_input_phase))
E_in_bottom = A_bottom*np.exp(1j*(bottom_input_phase))
input_E = np.array([E_in_top, E_in_bottom]) # this should be amplitude, not intensity
output_fields = np.zeros((num_points, 2), dtype=np.complex64)

i = 0
for inner_shift_value in theta_sweep:
    for outer_shift_value in phi_sweep:
        mzi = MZI(rho_1=0.5, rho_2=0.5, theta=inner_shift_value, phi=outer_shift_value)
        output_E = mzi.D_mzi@input_E
        output_fields[i,:] = output_E
        i+=1

output_intensity = np.abs(output_fields)**2
top_out_phase = np.zeros(num_points)
bottom_out_phase = np.zeros(num_points)
# [1:-1] for avoiding division by zero
top_out_phase[1:-1] = np.arctan2(np.imag(output_fields[1:-1, 0]),np.real(output_fields[1:-1, 0])) + np.pi
bottom_out_phase[1:-1] = np.arctan2(np.imag(output_fields[1:-1, 1]),np.real(output_fields[1:-1, 1])) + np.pi
# avoid discontinuity at end points
top_out_phase[0], top_out_phase[-1] = top_out_phase[1], top_out_phase[-2]
bottom_out_phase[0], bottom_out_phase[-1] = bottom_out_phase[1], bottom_out_phase[-2]

print(f"output_fields dimension({output_fields.shape}): \n{output_fields[:10]}")

print(f"E0 * e^(j*φ), E0 is real amplitude, φ is phase")
print(f"top input:    E0 = {A_top}, φ = {top_input_phase}")
print(f"bottom input: E0 = {A_bottom}, φ = {bottom_input_phase}\n")
print(f"total power:  {np.abs(E_in_top)**2+np.abs(E_in_bottom)**2}")
print(f"max and min Out1 power:  {np.max(output_intensity[:, 0]), np.min(output_intensity[:, 0])}")
print(f"max and min Out2 power:  {np.max(output_intensity[:, 1]), np.min(output_intensity[:, 1])}")
print(f"max power in Out1 occer when theta =  {theta_sweep[np.argmax(output_intensity[:, 0])]}")

dot_line_width = 0.8
fig, ax1 = plt.subplots()
ax2 = ax1.twinx()
# Customizing y-axis Ticks
ax1.xaxis.set_major_formatter(tck.FormatStrFormatter('%g $\pi$'))
ax1.xaxis.set_major_locator(tck.MultipleLocator(base=0.2))
ax1.yaxis.set_major_formatter(tck.FormatStrFormatter('%g'))
ax1.yaxis.set_major_locator(tck.MultipleLocator(base=0.1))
ax2.yaxis.set_major_formatter(tck.FormatStrFormatter('%g $\pi$'))
ax2.yaxis.set_major_locator(tck.MultipleLocator(base=0.25))
ax1.set_ylim([-0.04999967159447891, 1.0500001092475642]) # set y-axis limits for intensity
ax2.set_ylim([0.00, 2.00]) # set y-axis limits for phase
theta_sweep /=np.pi
top_out_phase /=np.pi
bottom_out_phase /=np.pi
print(f"top_out_phase ranges: {np.min(top_out_phase), np.max(top_out_phase)}, unit: π")
print(f"bottom_out_phase ranges: {np.min(bottom_out_phase), np.max(bottom_out_phase)}, unit: π")
ax1.plot(theta_sweep, output_intensity[:, 0], linestyle='-', color="red", label=f"top output intensity")
ax1.plot(theta_sweep, output_intensity[:, 1], linestyle='-', color="blue", label=f"bottom output intensity")
ax1.plot(theta_sweep, output_intensity[:, 0]+output_intensity[:, 1], linestyle='-', color="green", label=f"intensity sum")
ax2.plot(theta_sweep, top_out_phase, linestyle=':', color="purple", label=f"top output phase", linewidth=dot_line_width)
ax2.plot(theta_sweep, bottom_out_phase, linestyle=':', color="orange", label=f"bottom output phase", linewidth=dot_line_width)
# ax2.plot(theta_sweep, np.fmod(top_out_phase-bottom_out_phase, np.pi), linestyle='--', color="black", label=f"output phase diff", linewidth=dot_line_width)
plt.title(fr'relative P_outs of an MZI, E_in = [{A_top:.2f}exp(j{top_input_phase:.2f}) {A_bottom:.2f}exp(j{bottom_input_phase:.2f})]',fontsize=14)
ax1.set_xlabel('theta (rad)',fontsize=14)
ax1.set_ylabel('intensity',fontsize=14, weight='bold')
ax2.set_ylabel('abs. phase',fontsize=14, color='gray')
plt.xticks(fontsize=12)
plt.yticks(fontsize=10, color='gray')
# ax1.grid(linestyle='-', color='magenta')
ax2.grid(linestyle='--', color='gray')
# order = [0, 3, 1, 4, 2, 5]
# line_2d_list = [ax1.lines[0], ax1.lines[1], ax1.lines[2], ax2.lines[0], ax2.lines[1], ax2.lines[2]]
order = [0, 3, 1, 4, 2]
line_2d_list = [ax1.lines[0], ax1.lines[1], ax1.lines[2], ax2.lines[0], ax2.lines[1]]
label_list = [line2d.get_label() for line2d in line_2d_list]
print(label_list)
#[ax1.lines[0].get_label(), ax1.lines[1].get_label(), ax1.lines[2].get_label(), ax2.lines[0].get_label(), ax2.lines[1].get_label(), ax2.lines[2].get_label()]
plt.legend([line_2d_list[idx] for idx in order],
          [label_list[idx] for idx in order],
          loc='upper center', bbox_to_anchor=(0.5, -0.13),
          fancybox=True, shadow=True, ncol=3)
# Display the plot
# plt.savefig("Fiber_placement_ABCD.png")
print(f"ax1.get_ylim(): {ax1.get_ylim()}")
plt.tight_layout()
plt.show()

# %% Drop middle two lines to measure matrix similarity?
np.set_printoptions(precision=2)
np.random.seed(0)
n = 10
W = np.random.rand(n, n) # weight matrix
X = np.random.rand(n, 4) # data batch
# DropMat = np.eye(n)
DropMat = np.zeros((n, n))
DropMat[4, 4] = 1
DropMat[5, 5] = 1
Y = W
Y_drop = DropMat@W
print(f"X {X.shape}:\n{X}")
print(f"W {W.shape}:\n{W}")
print(f"Drop matrix {DropMat.shape}:\n{DropMat}")
print(f"Wi {Y.shape}:\n{Y}")
print(f"I x Wi {Y_drop.shape}:\n{Y_drop}")

# %% count MZI miniBokun
def count_MZI_miniBokun(N_:int):
    return (N_**2 + 10*N_ - 32)//8
    return N_//2*(N_//2+1)//2 + N_ - 4
    return np.sum(np.arange(N_//2,1,-1)) + N_ - 3
    # all three return statements are equivalent

print(count_MZI_miniBokun(8))
print(count_MZI_miniBokun(16))
print(count_MZI_miniBokun(32))
print(count_MZI_miniBokun(64))
# %% Mesh area estimation
import numpy as np
mzi_length = 300 # um, includes two phase shifters with length 135um each
wg_spacing = 60 # um, spacing between waveguides
wg_width = 0.5 # um, assuming 500um*220um waveguides

MESH_SIZES = np.array([8, 16, 32, 64])
COL_COUNTS = 2*MESH_SIZES-3 # Reck
COL_COUNTS = MESH_SIZES # Clements
COL_COUNTS = MESH_SIZES//2+1 # miniBokun
for topo in ['Reck', 'Clements', 'miniBokun']:
    print(f'Topology: {topo}')
    if topo=='Reck':
        COL_COUNTS = 2*MESH_SIZES-3 # Reck
    elif topo=='Clements':
        COL_COUNTS = MESH_SIZES # Clements
    elif topo=='miniBokun':
        COL_COUNTS = MESH_SIZES//2+1 # miniBokun
    else:
        print(f'Topology not understood.')
        break
    for (N, no_mzi_col) in zip(MESH_SIZES, COL_COUNTS):
        mesh_length = mzi_length*no_mzi_col
        mesh_width = wg_spacing*(N-1) + wg_width*N
        mesh_area = mesh_length*mesh_width
        # print(f"mesh_area (um^2): {mesh_area}")
        print(f"{N}x{N} mesh_area (mm^2): {mesh_area*1e-6:.3f}, with length (mm): {mesh_length*1e-3:.3f}")
    print('\n')



# %% Signal through Directional Coupler
# Directional Coupler
rho = 0.5 # coupling ratio for 3dB

dc = np.array([[np.sqrt(rho), 1j*np.sqrt(1-rho)],
               [1j*np.sqrt(1-rho), np.sqrt(rho)]])
# Signal
A_top = np.sqrt(0.9) # [0, 1] for relative amplitude
A_bottom = np.sqrt(1-A_top**2)
top_input_phase = 1.4 # [0, 2π)
bottom_input_phase = 1.4 # [0, 2π)

E_in_top = A_top*np.exp(1j*(top_input_phase))
E_in_bottom = A_bottom*np.exp(1j*(bottom_input_phase))
input_E = np.array([E_in_top, E_in_bottom]) # this is amplitude, not intensity

output_E = dc@input_E
print(f"input: {input_E}")
print(f"input intensity: {np.abs(input_E)**2}")
print(f"output: {output_E}")
print(f"output intensity: {np.abs(output_E)**2}")
print(f"output phase (unit: π): {np.angle(output_E)/np.pi}") # in pi units


# %% random predictor of "7" and "not 7" classes
P = 10 # number of 7s in the dataset
N = 90 # number of non-7s in the dataset
TP = P*0.5
TN = N*0.5
FP = P - TP
FN = N - TN

Precision = TP/(TP+FP)
Recall = TP/(TP+FN)
F1 = (Precision*Recall)/(Precision+Recall)*2
print(F1)


# %% Ideal 3dB Y-branch S-matrix
import numpy as np
S_matrix_y_branch_3db = np.array([[0, 1/np.sqrt(2), 1/np.sqrt(2)],
                     [1/np.sqrt(2), 0, 0],
                     [1/np.sqrt(2), 0, 0]])


A_top = np.sqrt(1) # [0, 1] for relative amplitude
A_bottom = np.sqrt(1-A_top**2)
top_input_phase = 0. # [0, 2π)
bottom_input_phase = 0. # [0, 2π)

E_in_top = A_top*np.exp(1j*(top_input_phase))
E_in_bottom = A_bottom*np.exp(1j*(bottom_input_phase))
input_E = np.array([E_in_top, 0, E_in_bottom]) # this should be amplitude, not intensity

output_E = S_matrix_y_branch_3db@input_E
print(f"input:\n{input_E}") 
print(f"output:\n{output_E}")
# %% Real part of complex (unitary) matrices
# Validation of page 3 of https://doi.org/10.1515/nanoph-2021-0521
def get_input_signal(dim=4, total_power=10, mu=0.4):
    # total power in mW, mu is the fraction of power used as the skip connected signal
    Ein = np.random.rand(dim) + 1j*np.zeros(dim)
    Ein = Ein/np.linalg.norm(Ein)*(np.sqrt(total_power*(1-mu)))
    print(f"Ein:\n{Ein}")
    print(f"normalized Ein power: {np.sum(np.abs(Ein)**2)}")
    return Ein
def get_skip_signal():
    return np.array([1, 0, 0, 0])

U = generate_random_unitary(4)
Ein = get_input_signal()

# print(f"U:\n{U}")
# print(f"Real part of U:\n{U.real}")
# Eout = U@Ein
# print(f"U@Ein:\n{Eout}")
# print(f"RE(U)@Ein:\n{U.real@Ein}")
# print(f"U@Ein intensity:\n{np.abs(Eout)**2}")
# print(f"RE(U)@Ein intensity:\n{np.abs(U.real@Ein)**2}")



# %% Draft
import numpy as np
m=2
n=3
num_type = np.int8
_B = np.array([[3, 5], [6, 7]], dtype=num_type, order="C")
placeholder = np.ones((4, 4), dtype=num_type, order="C")
# placeholder[m:m+2, m:m+2] = _B.copy()
placeholder[m, m] = _B.copy()[0,0]
placeholder[m, n] = _B.copy()[0,1]
placeholder[n, m] = _B.copy()[1,0]
placeholder[n, n] = _B.copy()[1,1]
print(placeholder)
# %%
