'''This module contains classes to implement Keras-style Models, which combine several NetworkLayers to simulate a full
optical neural network. Currently, only sequential models are supported, but more may be added in the future.'''

from typing import Dict, List
import numpy as np
from neuroptica.layers import NetworkLayer, OpticalMeshNetworkLayer

class BaseModel:
    '''Base class for all models'''

    def forward_pass(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def backward_pass(self, d_loss: np.ndarray) -> Dict[str, np.ndarray]:
        raise NotImplementedError

class Model(BaseModel):
    ''' Functional model class similar to the Keras model class, simulating an optical neural network with multiple layers
    '''

    def __init__(self):  # TODO
        pass

class Sequential(BaseModel):
    '''Feed-foward model class similar to the Keras Sequential() model class'''

    def __init__(self, layers: List[NetworkLayer]):
        '''
        Initialize the model
        :param layers: list of NetworkLayers contained in the optical neural network
        '''
        self.layers = layers
        self.input_cache = {}
        self.output_cache = {}
        for i, layer in enumerate(self.layers):
            layer.__name__ = "Layer{}_{}".format(i, layer.__class__.__name__)
            self.input_cache[layer.__name__] = []
            self.output_cache[layer.__name__] = []

    def __repr__(self):
        return "<Sequential Model: {}>".format([layer.__name__ for layer in self.layers])

    def get_transformation_matrix(self):
        trf_matrix = []
        for layer in self.layers:
            if isinstance(layer, OpticalMeshNetworkLayer):
                trf_matrix.append(layer.mesh.get_transfer_matrix())
        return trf_matrix

    def forward_pass(self, X: np.ndarray, cache_fields=False, use_partial_vectors=False) -> np.ndarray:
        '''
        Propagate an input field throughout the entire network
        :param X: input electric fields
        :param cache_fields: if true, fields will be cached internally
        :param use_partial_vectors: if true, use the partial vectors method to speed up transfer matrix computation
        :return: output electric fields (to be fed into a loss function)
        '''
        X_out = X
        #print(X_out.shape)
        for layer in self.layers:
            if isinstance(layer, OpticalMeshNetworkLayer):
                X_out = layer.forward_pass(X_out)
                # print(layer.mesh.get_transfer_matrix())
                # print(list(layer.mesh.all_tunable_params()))
                # print(X_out.T)
            else:
                X_out = layer.forward_pass(X_out)
        #print(X_out.shape)
        return X_out

    def backward_pass(self, d_loss: np.ndarray, cache_fields=False, use_partial_vectors=False) -> Dict[str, np.ndarray]:
        '''
        Returns the gradients for each layer resulting from backpropagating from derivative loss function d_loss
        :param d_loss: derivative of the loss function of the outputs
        :param cache_fields: if true, fields will be cached internally
        :param use_partial_vectors: if true, use the partial vectors method to speed up transfer matrix computation
        :return: dictionary of {layer: gradients}
        '''
        backprop_signal = d_loss
        gradients = {"output": d_loss}
        for layer in reversed(self.layers):
            if isinstance(layer, OpticalMeshNetworkLayer):
                backprop_signal = layer.backward_pass(backprop_signal)
            else:
                backprop_signal = layer.backward_pass(backprop_signal)

            gradients[layer.__name__] = backprop_signal
        return gradients
    
    def get_all_phases(self):
        phases = []
        for layer in self.layers:
            if hasattr(layer, 'mesh'):
                phases.append([x for x in layer.mesh.all_tunable_params()])
        return phases

    def get_all_losses(self):
        losses = []
        for layer in self.layers:
            if hasattr(layer, 'mesh'):
                losses.append([x for x in layer.mesh.all_losses()])
        return losses

    def set_all_phases_uncerts_losses(self, Phases, phase_uncert_theta=0, phase_uncert_phi=0, loss_dB=0, loss_diff=0):
        phase_idx = 0
        # print("Set all phase uncerts", loss_dB)
        for layer in self.layers:
            if hasattr(layer, 'mesh'):
                layer.set_phases_uncert_loss(Phases[phase_idx], phase_uncert_theta, phase_uncert_phi, loss_dB, loss_diff)
                phase_idx += 1
