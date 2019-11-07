
"""
Testing the inverted reck mesh, single MZI
Author: Simon Geoffroy-Gagnon
Edit: 07.11.2019
"""
import pandas as pd
from sklearn.datasets.samples_generator import make_blobs
# import scatter_matrix_plotter as pb
# import iris_fourth_flower as ir
import sys
import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import pandas as pd
import time
# Set random seed to always get same data
rng = 6
random.seed(rng)

sys.path.append('/home/simon/Documents/neuroptica')
import neuroptica as neu

def blob_maker(targets=4, features=4, nsamples=10000,
               cluster_std=.1):

    # generate 2d classification dataset
    X, y = make_blobs(n_samples=nsamples, centers=targets,
                      n_features=features,
                      cluster_std=cluster_std,
                      center_box=(0, 1), random_state=rng, shuffle=False)
    ohe_labels = pd.get_dummies(y).values
    x, xt, y, yt = train_test_split(X, ohe_labels, test_size=0.2)

    return x, y, xt, yt

def printf(format, *args):
    sys.stdout.write(format % args)

# Number of input features?
N = 4
theta =  np.pi/2
phi =  np.pi/2
loss = 0 # dB

# X, y, Xt, yt, *_ = mnist.get_data([1,3,6,7])
X, y, Xt, yt = blob_maker(targets=N, features=N, nsamples=5000)

# Normalize inputs
X = (X - np.min(X))/(np.max(X) - np.min(X))
# pb.plot_blobs(X,y)

losses_dB = [0.25*x for x in range(1)]# in dB
# losses_dB = [0, 0.5]
iterations = 1 # number of times to retry same loss
phase_uncerts = [0] #np.linspace(0, 1.5, 31)

Folder = 'Accuracy_vs_phaseUncert_300iter_100%acc/'
setup = ['Reck','invReck','Reck+invReck','Reck+DMM+invReck']

accuracy_per_model = []
for idx in [0]: 
    print(f'model{idx}')
    accuracy = []
    for loss in losses_dB:
        print(f'loss = {loss} dB')

        # First train the network with 0 phase noise
        phase_uncert = 0 
        if idx == 0:
            model = neu.Sequential([
                neu.ReckLayer(N, include_phase_shifter_layer=False, thetas=[None]*int(N*(N-1)/2), phis=[None]*int(N*(N-1)/2), loss=loss, phase_uncert=phase_uncert),

                # neu.AddMask(2*N),
                # neu.DMM_layer(2*N, thetas=[None]*int(N), phis=[None]*int(N)),
                # neu.DropMask(N=2*N, keep_ports=range(0, 2*N, 2)), # makes the DMM thetas pi
                # neu.DropMask(N=2*N, keep_ports=range(1, 2*N+1, 2)), # makes the DMM thetas 0 or 2*pi

                # neu.flipped_ReckLayer(N, include_phase_shifter_layer=False, thetas=[None]*int(N*(N-1)/2), phis=[None]*int(N*(N-1)/2), loss=loss),

                neu.Activation(neu.AbsSquared(N)), # photodetector measurement
            ])
        if idx == 1:
            model = neu.Sequential([
                # neu.ReckLayer(N, include_phase_shifter_layer=False, thetas=[None]*int(N*(N-1)/2), phis=[None]*int(N*(N-1)/2), loss=loss, phase_uncert=phase_uncert),

                # neu.AddMask(2*N),
                # neu.DMM_layer(2*N, thetas=[None]*int(N), phis=[None]*int(N)),
                # neu.DropMask(N=2*N, keep_ports=range(0, 2*N, 2)), # makes the DMM thetas pi
                # neu.DropMask(N=2*N, keep_ports=range(1, 2*N+1, 2)), # makes the DMM thetas 0 or 2*pi

                neu.flipped_ReckLayer(N, include_phase_shifter_layer=False, thetas=[None]*int(N*(N-1)/2), phis=[None]*int(N*(N-1)/2), loss=loss),

                neu.Activation(neu.AbsSquared(N)), # photodetector measurement
            ])
        if idx == 2:
            model = neu.Sequential([
                neu.ReckLayer(N, include_phase_shifter_layer=False, thetas=[None]*int(N*(N-1)/2), phis=[None]*int(N*(N-1)/2), loss=loss, phase_uncert=phase_uncert),

                # neu.AddMask(2*N),
                # neu.DMM_layer(2*N, thetas=[None]*int(N), phis=[None]*int(N)),
                # neu.DropMask(N=2*N, keep_ports=range(0, 2*N, 2)), # makes the DMM thetas pi
                # neu.DropMask(N=2*N, keep_ports=range(1, 2*N+1, 2)), # makes the DMM thetas 0 or 2*pi

                neu.flipped_ReckLayer(N, include_phase_shifter_layer=False, thetas=[None]*int(N*(N-1)/2), phis=[None]*int(N*(N-1)/2), loss=loss),

                neu.Activation(neu.AbsSquared(N)), # photodetector measurement
            ])
        if idx == 3:
            model = neu.Sequential([
                neu.ReckLayer(N, include_phase_shifter_layer=False, thetas=[None]*int(N*(N-1)/2), phis=[None]*int(N*(N-1)/2), loss=loss, phase_uncert=phase_uncert),

                neu.AddMask(2*N),
                neu.DMM_layer(2*N, thetas=[None]*int(N), phis=[None]*int(N)),
                # neu.DropMask(N=2*N, keep_ports=range(0, 2*N, 2)), # makes the DMM thetas pi
                neu.DropMask(N=2*N, keep_ports=range(1, 2*N+1, 2)), # makes the DMM thetas 0 or 2*pi

                neu.flipped_ReckLayer(N, include_phase_shifter_layer=False, thetas=[None]*int(N*(N-1)/2), phis=[None]*int(N*(N-1)/2), loss=loss),

                neu.Activation(neu.AbsSquared(N)), # photodetector measurement
            ])

        # initialize the ADAM optimizer and fit the ONN to the training data
        optimizer = neu.InSituAdam(model, neu.MeanSquaredError, step_size=0.001)
        losses, trn_accuracy, val_accuracy = optimizer.fit(X.T, y.T, Xt.T, yt.T, epochs=70, batch_size=2**5, show_progress=True)

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
        plt.title(f'Gradient Descent, Final Validation Accuracy: {val_accuracy[-1]:.2f}')
        plt.ylim([0, 100])
        plt.savefig(f'{Folder}{setup[idx]}_loss={loss}_uncert={phase_uncert:.3f}.pdf')
        plt.clf()


        # save a txt file containing the loss, trn acc, val acc, in case i want to replot it using matlab
        np.savetxt(f'{Folder}{setup[idx]}_loss={loss}_uncert={phase_uncert:.3f}.txt',np.array([losses, trn_accuracy, val_accuracy]).T, delimiter=',', fmt='%.4f')

        # Create phase array
        phases = []
        for layer in model.layers:
            if hasattr(layer, 'mesh'):
                phases.append([x for x in layer.mesh.all_tunable_params()])

        # separate phase array into theta and phi
        thetas = [item for sublist in phases for (item, _) in sublist]
        phis = [item for sublist in phases for (_, item) in sublist]

        # Now calculate the accuracy when adding phase noise and/or mzi loss
        acc_array = []
        for phase_uncert in phase_uncerts:
            if idx == 0:
                model = neu.Sequential([
                    neu.ReckLayer(N, include_phase_shifter_layer=False, thetas=thetas, phis=phis, loss=loss, phase_uncert=phase_uncert),

                    # neu.AddMask(2*N),
                    # neu.DMM_layer(2*N, thetas=[None]*int(N), phis=[None]*int(N)),
                    # neu.DropMask(N=2*N, keep_ports=range(0, 2*N, 2)), # makes the DMM thetas pi
                    # neu.DropMask(N=2*N, keep_ports=range(1, 2*N+1, 2)), # makes the DMM thetas 0 or 2*pi

                    # neu.flipped_ReckLayer(N, include_phase_shifter_layer=False, thetas=[None]*int(N*(N-1)/2), phis=[None]*int(N*(N-1)/2), loss=loss),

                    neu.Activation(neu.AbsSquared(N)), # photodetector measurement
                ])
            if idx == 1:
                model = neu.Sequential([
                    # neu.ReckLayer(N, include_phase_shifter_layer=False, thetas=[None]*int(N*(N-1)/2), phis=[None]*int(N*(N-1)/2), loss=loss, phase_uncert=phase_uncert),

                    # neu.AddMask(2*N),
                    # neu.DMM_layer(2*N, thetas=[None]*int(N), phis=[None]*int(N)),
                    # neu.DropMask(N=2*N, keep_ports=range(0, 2*N, 2)), # makes the DMM thetas pi
                    # neu.DropMask(N=2*N, keep_ports=range(1, 2*N+1, 2)), # makes the DMM thetas 0 or 2*pi

                    neu.flipped_ReckLayer(N, include_phase_shifter_layer=False, thetas=thetas, phis=phis, loss=loss, phase_uncert=phase_uncert),

                    neu.Activation(neu.AbsSquared(N)), # photodetector measurement
                ])
            if idx == 2:
                model = neu.Sequential([
                    neu.ReckLayer(N, include_phase_shifter_layer=False, thetas=thetas[:int(N*(N-1)/2)], phis=phis[:int(N*(N-1)/2)], loss=loss, phase_uncert=phase_uncert),

                    # neu.AddMask(2*N),
                    # neu.DMM_layer(2*N, thetas=[None]*int(N), phis=[None]*int(N)),
                    # neu.DropMask(N=2*N, keep_ports=range(0, 2*N, 2)), # makes the DMM thetas pi
                    # neu.DropMask(N=2*N, keep_ports=range(1, 2*N+1, 2)), # makes the DMM thetas 0 or 2*pi

                    neu.flipped_ReckLayer(N, include_phase_shifter_layer=False, thetas=thetas[int(N*(N-1)/2):], phis=phis[int(N*(N-1)/2):], loss=loss, phase_uncert=phase_uncert),

                    neu.Activation(neu.AbsSquared(N)), # photodetector measurement
                ])
            if idx == 3:
                model = neu.Sequential([
                    neu.ReckLayer(N, include_phase_shifter_layer=False, thetas=thetas[:int(N*(N-1)/2)], phis=phis[:int(N*(N-1)/2)], loss=loss, phase_uncert=phase_uncert),

                    neu.AddMask(2*N),
                    neu.DMM_layer(2*N, thetas=thetas[int(N*(N-1)/2):int(N*(N-1)/2)+int(N)], phis=phis[int(N*(N-1)/2):int(N*(N-1)/2)+int(N)], phase_uncert=phase_uncert, loss=loss),
                    # neu.DropMask(N=2*N, keep_ports=range(0, 2*N, 2)), # makes the DMM thetas pi
                    neu.DropMask(N=2*N, keep_ports=range(1, 2*N+1, 2)), # makes the DMM thetas 0 or 2*pi

                    neu.flipped_ReckLayer(N, include_phase_shifter_layer=False, thetas=thetas[-int(N*(N-1)/2):], phis=phis[-int(N*(N-1)/2):], loss=loss, phase_uncert=phase_uncert),

                    neu.Activation(neu.AbsSquared(N)), # photodetector measurement
                ])
            acc = []    

            # calculate validation accuracy. Since phase is shifted randomly after iteration, this is good
            for _ in range(300):
                Y_hat = model.forward_pass(Xt.T)
                pred = np.array([np.argmax(yhat) for yhat in Y_hat.T])
                gt = np.array([np.argmax(tru) for tru in yt])
                acc.append(np.sum(pred == gt)/len(Xt)*100)
            acc_array.append(np.mean(acc))
        accuracy.append(acc_array)

    # save the accuracies of the current model. will be a 2D array for accuracies at every loss_dB and phase_uncert
    np.savetxt(f'{Folder}accuracy_{setup[idx]}.txt', np.array(accuracy).T, delimiter=',', fmt='%.3f')
    accuracy_per_model.append(accuracy)

# save loss_dB and phase_uncert too
np.savetxt(f'{Folder}LossdB.txt',losses_dB, delimiter=',',fmt='%.3f')
np.savetxt(f'{Folder}PhaseUncert.txt',phase_uncerts, delimiter=',',fmt='%.3f')


if 0:
    # print(accuracy)
    lab = [f'Loss = {loss}' for loss in losses_dB]
    # print(lab)
    plt.plot(phase_uncerts, np.array(accuracy).T)
    plt.legend(lab, loc='best')
    plt.ylabel('Accuracy')
    plt.xlabel("Phase Uncertainty ($\sigma_\\theta = \sigma_\phi$)")
    plt.title(f"Phase Uncertainty Vs Accuracy\n{setup[idx]}")
    plt.savefig(f'{setup[idx]}_accVSuncert.pdf')
    plt.clf()

    np.savetxt(f'{setup[idx]}_acc.txt', np.array(accuracy).T, delimiter=",", fmt='%.4f')
    np.savetxt("acc_vs_uncert+loss/losses_dB.txt",losses_dB, delimiter=',', fmt='%.4f')
    np.savetxt("acc_vs_uncert+loss/phase_uncert.txt", phase_uncerts, delimiter=',', fmt='%.4f')
    # for i in [1]:
        # continue 

               
        # thetas.append([item for sublist in phases for (item, _) in sublist])
        #    phis.append([item for sublist in phases for (_, item) in sublist])

        # Get final accuracy
        

        # for layer in model.layers:
        #     if hasattr(layer, 'mesh'):
        #         phases.append([x for x in layer.mesh.all_tunable_params()])
        #Reck_layer_idx = [f'Reck, MZI_{mzi}' for mzi in range(int(N*(N-1)/2)+N)]
        #DMM_layer_idx = [f'DMM, MZI_{mzi}' for mzi in range(int((N*(N-1))/2),int(N*(N-1)/2+N))]
        #Reck_H_layer_idx = [f'Reck_H, MZI_{mzi}' for mzi in range(int(N*(N-1)/2+N),int(N*(N-1)+2*N))]
        ## df = pd.DataFrame(np.array([thetas, phis]).T, columns=['Theta','Phi'],  index=Reck_layer_idx)
        ## df = pd.DataFrame(np.array([thetas, phis]).T, columns=['Theta','Phi'])
        ## df = pd.DataFrame(np.array([thetas, phis]).T, columns=['Theta','Phi'], index=Reck_layer_idx+DMM_layer_idx+Reck_H_layer_idx)

        ## psrint(df.round(2))
        #acc_mean.append(val_accuracy[-1])
        ## print(D)
        #np.savetxt("X.csv", X, delimiter=",")
        #np.savetxt("y.csv", y, delimiter=",")
        #np.savetxt("yt.csv", yt, delimiter=",")
        #np.savetxt("Xt.csv", Xt, delimiter=",")
    if 0:            
        mean_t = np.array(thetas).mean(axis=0)
        std_t = np.array(thetas).std(axis=0)
        mean_p = np.array(phis).mean(axis=0)
        std_p = np.array(phis).std(axis=0)
        print(f"loss = {loss}")
        print("Mean Accuracy: {}".format(np.array(acc_mean).mean()))
        print("thetas, mean:")
        print(mean_t)
        print("thetas, std:")
        print(std_t)
        print("phis, mean:")
        print(mean_p)
        print("phis, std:")
        print(std_p)
        print('')
    # np.savetxt("Phases.txt", phases, delimiter=",")
        # print(D)
