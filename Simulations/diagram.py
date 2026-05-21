#%% imports
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rcParams['axes.linewidth'] = 1
np.set_printoptions(precision=3, suppress=True, linewidth=np.inf)
legend_size = 20
text_size = 20
tick_size = 15
#%% Pruning process and accuracy monitoring
fig = plt.figure(figsize=(8, 6))
ax= plt.gca()
# plt.title(f'Sensitivity Analysis on Phase Error & Loss Tolerance, beta=1.2', fontsize=15)
# plt.xlabel(f'Topology', fontsize=text_size)
plt.ylabel(f'Accuracy (%)', fontsize=text_size)
plt.xlabel(f'Number of Pruned MZI Layers', fontsize=text_size)
topologies = ["0\n(Clements)", 1, 2, 3, 4, 5, "6", "\n(MiniBokun)"]
accuracies = [71.828, 71.946, 71.89, 71.934, 72.024, 71.356, 70.176, 71.518]
ax.plot(topologies, accuracies, linestyle='-', color="red", marker='o', label=r"$10{\times}10$ ONN")
ax.text(3.75, 70.3, "(triangle)", fontsize=tick_size-2, fontweight="bold")
# ax.text(3.7, 71.0, "(triangle)", fontsize=tick_size-2, fontweight="bold")

ax.tick_params(axis='y', which='major', labelsize=tick_size)
ax.set_xticklabels(topologies, fontsize=tick_size-2, fontweight='bold')
# ax.get_xticklabels()[-2].set_rotation(10)
# ax.get_xticklabels()[-1].set_rotation(10)
plt.grid(True)
plt.legend(loc='upper right',prop={'size': legend_size-2})
plt.show()
plt.close()

#%% FoM Bar Plots
fig = plt.figure(figsize=(8, 6))
# plt.title(f'Sensitivity Analysis on Phase Error & Loss Tolerance, beta=1.2', fontsize=15)
# plt.xlabel(f'Topology', fontsize=text_size)
plt.ylabel(f'PT FoM (rad²)', fontsize=text_size)
# plt.ylabel(f'LPU FoM (rad•dB)', fontsize=text_size)

names = ["Reck", "Clements", "MiniBokun"]
x_range_pos = np.arange(1, 4, 1) # total num of bars

#              [Reck, Clements, MiniBokun]
y_MNIST_PT_N_8 = np.array([0.046,0.090,0.148])
y_MNIST_LPU_N_8 = np.array([0.133,0.170,0.227])
y_MNIST_PT_N_16 = np.array([0.026,0.054,0.080])
y_MNIST_LPU_N_16 = np.array([0.028,0.135,0.110])

y_CIFAR_PT_N_8 = np.array([0.030, 0.078, 0.121])
y_CIFAR_LPU_N_8 = np.array([0.068, 0.148, 0.164])
y_CIFAR_PT_N_16 = np.array([0.011, 0.027, 0.054])
y_CIFAR_LPU_N_16 = np.array([0.027, 0.057, 0.125])

all_PT_data = np.vstack((y_MNIST_PT_N_8,y_MNIST_PT_N_16,y_CIFAR_PT_N_8, y_CIFAR_PT_N_16))
all_LPU_data = np.vstack((y_MNIST_LPU_N_8,y_MNIST_LPU_N_16,y_CIFAR_LPU_N_8, y_CIFAR_LPU_N_16))
print(np.average((all_PT_data[:, 2] - all_PT_data[:, 1])/all_PT_data[:, 1])) # avg improvement to clements
print(np.average((all_LPU_data[:, 2] - all_LPU_data[:, 1])/all_LPU_data[:, 1])) # avg improvement to clements



# all_N8_PT = np.concatenate((y_MNIST_PT_N_8, y_CIFAR_PT_N_8))
# all_N16_PT = np.concatenate((y_MNIST_PT_N_16, y_CIFAR_PT_N_16))
# print(np.average((all_N8_PT - all_N16_PT)/all_N8_PT))
# all_N8_LPU = np.concatenate((y_MNIST_LPU_N_8, y_CIFAR_LPU_N_8))
# all_N16_LPU = np.concatenate((y_MNIST_LPU_N_16, y_CIFAR_LPU_N_16))
# print(np.average((all_N8_LPU - all_N16_LPU)/all_N8_LPU))


data = [y_MNIST_PT_N_8, y_CIFAR_PT_N_8]

width=0.4
# overlap_bars = plt.bar(x_range_pos-width, y_overlap, width=width, align='center', color='red', label="overlap")
MNIST_bars = plt.bar(x_range_pos-width, data[0], width=width, align='edge', color='orange', label="MNIST")
CIFAR_bars = plt.bar(x_range_pos, data[1], width=width, align='edge', color='green', label="CIFAR-10")

ax = plt.gca()
ax.bar_label(MNIST_bars, fmt='%.3f', color='black', rotation=0, padding=3, fontsize=text_size, fontweight='regular')
ax.bar_label(CIFAR_bars,  fmt='%.3f', color='black', rotation=0, padding=3, fontsize=text_size, fontweight='regular')
ax.set_xticks(x_range_pos, labelsize=tick_size)
max_mse = np.max(data)
# ax.set_ylim([0, max_mse + 0.05])
ax.set_ylim([0, 0.25])
# ax.set_yticks(np.arange(0, 40, 5))
ax.tick_params(axis='y', which='major', labelsize=tick_size)
ax.set_xticklabels(names, fontsize=text_size, fontweight='regular')
plt.legend(loc='upper left',prop={'size': legend_size})
fig.tight_layout()
# fig.savefig(f"./SenAnalysis_.png")
plt.show()
plt.close()


# %% Box and Whisker plot for Expressibility
fig, ax = plt.subplots(4, 4, figsize=(22, 14))

colors = {
    'clements': "lightcyan",
    'triangle': "mistyrose",
    'miniBokun': "palegreen",
    'rand_unitary': "lavender"
}
topology_Name = {
    'clements': "Clements",
    'triangle': "Triangle",
    'miniBokun': "MiniBokun",
    'rand_unitary': "RandUnitary"
}

# topo = "triangle" # "clements" or "miniBokun"
for row, port_choice in enumerate([0, 3, 5, -1]):
    for col, topo in enumerate(["clements", "triangle", "miniBokun", "rand_unitary"]):
        if topo == "rand_unitary":
            loaded_statistics = np.load(f'./Analysis/iris_augment/EXPRESSIVITY_STUDY/EXP_miniBokun_port_{port_choice}_statistics_RANDOM_UNITARY.npy')
        else:
            loaded_statistics = np.load(f'./Analysis/iris_augment/EXPRESSIVITY_STUDY/EXP_{topo}_port_{port_choice}_statistics.npy')
        # rows: [min, q1, median, q3, max, mean]
        print(f"==========================================port {port_choice}, {topo}==========================================")
        print(max(loaded_statistics[-2]) - min(loaded_statistics[-2]))
        # continue
        port_min = loaded_statistics[0]
        port_q1 = loaded_statistics[1]
        port_median = loaded_statistics[2]
        port_q3 = loaded_statistics[3]
        port_max = loaded_statistics[4]
        port_mean = loaded_statistics[5]

        boxes = []
        for i in range(8):
            IQR = port_q3[i]-port_q1[i]
            boxes.append(
                {
                    'label' : f"{i}",
                    'whislo': port_min[i],    # Bottom whisker position
                    'q1'    : port_q1[i],    # First quartile (25th percentile)
                    'med'   : port_median[i],    # Median         (50th percentile)
                    'q3'    : port_q3[i],    # Third quartile (75th percentile)
                    'whishi': port_max[i],    # Top whisker position
                    'fliers': []        # Outliers
                }
            )
        box_plot = ax[row, col].bxp(boxes,
                        showfliers=True,
                        patch_artist=True,
                        medianprops=dict(color="orange"),
                        boxprops=dict(facecolor=colors[topo], edgecolor="cornflowerblue"),
                        whiskerprops=dict(color="cornflowerblue"),
                        capprops=dict(color="red"),
                        # positions=np.arange(loaded_statistics.shape[1])+col*0.25, widths=0.2
                        )
        # compute s.d. of the max
        std_dev = np.std(port_max)
        # label the s.d. on the plot
        ax[row, col].text(8.5, 3, r"std$_{max}$:"+f"{std_dev:.2f}", fontsize=text_size, color="black", rotation=-90)
        ax[row, col].minorticks_on()
        ax[row, col].tick_params(axis='x', which='major', labelsize=tick_size, rotation=0)
        ax[row, col].tick_params(axis='y', which='major', labelsize=tick_size)
        ax[row, col].tick_params(axis='x', which='minor', labelsize=tick_size, bottom=False)
        if row == 0: ax[row, col].set_title(f'{topology_Name[topo]}',fontsize=text_size)
        if row == 3: ax[row, col].set_xlabel(f"Port Number", fontsize=text_size)
        if col == 0: ax[row, col].set_ylabel(r"$P_{out}$ (mW)", fontsize=text_size)
        ax[row, col].get_xticklabels()[port_choice].set_color("red")
        ax[row, col].get_xticklabels()[port_choice].set_weight("bold")
        if port_choice == -1:
            for i in range(8):
                ax[row, col].get_xticklabels()[i].set_color("red")
                ax[row, col].get_xticklabels()[i].set_weight("bold")

        ax[row, col].grid(True, which='both', linestyle='--', linewidth=0.5)
plt.show()
# plt.savefig("boxplot.png")
plt.close()

# %% Accuracy plot
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["figure.figsize"] = [5.2, 4]
plt.rcParams["figure.autolayout"] = True
plt.rcParams['axes.linewidth'] = 2
plt.rcParams['lines.linewidth'] = 2
# set font size for axes labels
plt.rcParams['axes.labelsize'] = 14
# set font size for legend
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.dpi'] = 600

N = [8, 16, 32, 64]
dataset = "CIFAR"
if dataset == "MNIST":
    # MNIST
    accuracy_clements = [70.88, 75.87, 77.58, 78.40]
    accuracy_reck = [71.30, 74.32, 77.43, 78.51]
    accuracy_minibokun = [71.47, 75.26, 77.16, 77.39]

    accuracy_clements_10 = [71.04, 75.16, 75.97, 75.65]
    accuracy_reck_10 = [71.86, 74.62, 75.82, 75.72]
    accuracy_minibokun_10 = [70.65, 74.94, 76.33, 75.72]
elif dataset == "CIFAR":
    # CIFAR-10
    accuracy_clements = [72.32, 73.35, 74.19, 74.79]
    accuracy_reck = [71.92, 73.31, 74.22, 74.75]
    accuracy_minibokun = [72.18, 73.04, 73.39, 73.56]

    accuracy_clements_10 = [72.71, 72.67, 72.04, 71.52]
    accuracy_reck_10 = [72.10, 72.56, 71.93, 71.59]
    accuracy_minibokun_10 = [72.38, 72.24, 72.45, 71.80]

plt.plot(N, accuracy_clements, label = "Clements - Fixed optical input power per channel")
plt.plot(N, accuracy_reck, label = "Reck - Fixed optical input power per channel")
plt.plot(N, accuracy_minibokun, label = "MiniBokun - Fixed optical input power per channel")

plt.plot(N, accuracy_clements_10, label = "Clements - Constant total optical input power", linestyle='--')
plt.plot(N, accuracy_reck_10, label = "Reck - Constant total optical input power", linestyle='--')
plt.plot(N, accuracy_minibokun_10, label = "MiniBokun - Constant total optical input power", linestyle='--')

plt.ylim([70, 80])
plt.yticks([70, 72, 74, 76, 78, 80])
#tick thickness
plt.tick_params(axis='both', which='major', labelsize=12, width=2)

plt.xlabel("N")
plt.ylabel("Test Accuracy [%]")

if dataset == "CIFAR": plt.legend(loc = 'best')
# plt.tight_layout()
plt.savefig(f"Fig_acc_vs_N_{dataset}.png", dpi = 600)
plt.show()
plt.close()


# %% #MZI plot
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["figure.figsize"] = [5.2, 4]
plt.rcParams["figure.autolayout"] = True
plt.rcParams['axes.linewidth'] = 2
plt.rcParams['lines.linewidth'] = 2
# set font size for axes labels
plt.rcParams['axes.labelsize'] = 14
# set font size for legend
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.dpi'] = 600

N = np.arange(8, 65, 2)
clements_no_mzi = (N**2 - N)//2
miniBokun_no_mzi = (N**2 + 10*N - 32)//8
clements_path_diff = N//2
reck_path_diff = N - 2
miniBokun_path_diff = np.ones(len(N))*2

plt.plot(N, clements_no_mzi, label = "Clements/Reck - number of MZIs")
plt.plot(N, miniBokun_no_mzi, label = "MiniBokun - number of MZIs", color='green')
# plt.plot(N, clements_path_diff, label = "Clements - path length difference", linestyle='--')
# plt.plot(N, reck_path_diff, label = "Reck - path length difference", linestyle='--')
# plt.plot(N, miniBokun_path_diff, label = "MiniBokun - path length difference", linestyle='--')

# plt.ylim([0, 250])
# plt.yticks([70, 72, 74, 76, 78, 80])
plt.ylabel("Number of MZIs")
plt.xlabel("N")
plt.tick_params(axis='both', which='major', labelsize=12, width=2)
plt.minorticks_on()
plt.grid(True, which='both', linestyle='--', linewidth=0.5)

plt.legend(loc = 'best')
# plt.tight_layout()
# plt.savefig(f"MZI_counts.png", dpi = 600)
plt.show()
plt.close()

# %%

