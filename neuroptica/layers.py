'''The layers submodule contains functionality for implementing a logical "layer" in the simulated optical neural
network. The API for this module is based loosely on Keras.'''

import numpy as np

from neuroptica.component_layers import MZILayer, MZILayer_H, OpticalMesh, PhaseShifterLayer
from neuroptica.nonlinearities import Nonlinearity
from neuroptica.settings import NP_COMPLEX


class NetworkLayer:
    '''Represents a logical layer in a simulated optical neural network. A NetworkLayer is different from a
    ComponentLayer, but it may contain a ComponentLayer or an OpticalMesh to compute the forward and backward logic.'''

    def __init__(self, input_size: int, output_size: int, initializer=None):
        '''
        Initialize the NetworkLayer
        :param input_size: number of input ports
        :param output_size: number of output ports (usually the same as input_size, unless DropMask is used)
        :param initializer: optional initializer method (WIP)
        '''
        self.input_size = input_size
        self.output_size = output_size
        self.initializer = initializer
        self.input_prev: np.ndarray = None
        self.output_prev: np.ndarray = None

    def forward_pass(self, X: np.ndarray) -> np.ndarray:
        '''
        Compute the forward pass of input fields into the network layer
        :param X: input fields to the NetworkLayer
        :return: transformed output fields to feed into the next layer of the ONN
        '''
        raise NotImplementedError('forward_pass() must be overridden in child class!')

    def backward_pass(self, delta: np.ndarray) -> np.ndarray:
        '''
        Compute the backward (adjoint) pass, given a backward-propagating field shined into the layer from the outputs
        :param delta: backward-propagating field shining into the NetworkLayer outputs
        :return: transformed "input" fields to feed to the previous layer of the ONN
        '''
        raise NotImplementedError('backward_pass() must be overridden in child class!')

class AddMask(NetworkLayer):
    '''interleaves 0s beween existing ports (essentially adding extra waveguides for the DMM section)'''
    def __init__(self, N: int):
        self.ports = list(range(N))
        super().__init__(N, len(self.ports))

    def forward_pass(self, X: np.ndarray):
        B = np.zeros_like(X, dtype=NP_COMPLEX)
        C = np.empty((X.shape[0]+B.shape[0],X.shape[1]), dtype=NP_COMPLEX)
        C[::2,:] = X
        C[1::2,:] = B
        return C

    def backward_pass(self, delta: np.ndarray) -> np.ndarray:
        n_features, n_samples = delta.shape
        delta_back = np.zeros((int(self.input_size/2), n_samples), dtype=NP_COMPLEX)
        for i in range(int(n_features/2)):
            delta_back[self.ports[i]] = delta[i*2]
        return delta_back

class DropMask(NetworkLayer):
    '''Drop specified ports entirely, reducing the size of the network for the next layer.'''

    def __init__(self, N: int, keep_ports=None, drop_ports=None):
        '''
        :param N: number of input ports to the DropMask layer
        :param keep_ports: list or iterable of which ports to keep (drop_ports must be None if keep_ports is specified)
        :param drop_ports: list or iterable of which ports to drop (keep_ports must be None if drop_ports is specified)
        '''
        if (keep_ports is not None and drop_ports is not None) or (keep_ports is None and drop_ports is None):
            raise ValueError("specify exactly one of keep_ports or drop_ports")
        if keep_ports:
            if isinstance(keep_ports, range):
                keep_ports = list(keep_ports)
            elif isinstance(keep_ports, int):
                keep_ports = [keep_ports]
            self.ports = keep_ports
        elif drop_ports:
            ports = list(range(N))
            for port in drop_ports:
                ports.remove(port)
            self.ports = ports
        super().__init__(N, len(self.ports))

    def forward_pass(self, X: np.ndarray):
        return X[self.ports]

    def backward_pass(self, delta: np.ndarray) -> np.ndarray:

        n_features, n_samples = delta.shape
        delta_back = np.zeros((self.input_size, n_samples), dtype=NP_COMPLEX)
        for i in range(n_features):
            delta_back[self.ports[i]] = delta[i]
        return delta_back

class StaticMatrix(NetworkLayer):
    '''Multiplies inputs by a static matrix (this is an aphysical layer)'''

    # TODO: make less hacky

    def __init__(self, matrix: np.ndarray):
        '''
        :param matrix: matrix to multiply inputs by
        '''
        N_out, N_in = matrix.shape
        super().__init__(N_in, N_out)
        self.matrix = matrix

    def forward_pass(self, X: np.ndarray):
        return self.matrix @ X

    def backward_pass(self, delta: np.ndarray):
        return self.matrix.T @ delta

class Activation(NetworkLayer):
    '''Represents a (nonlinear) activation layer. Note that in this layer, the usage of X and Z are reversed!
    (Z is input, X is output, input for next linear layer)
    '''

    def __init__(self, nonlinearity: Nonlinearity):
        '''
        Initialize the activation layer
        :param nonlinearity: a Nonlinearity instance
        '''
        super().__init__(nonlinearity.N, nonlinearity.N)
        self.nonlinearity = nonlinearity

    def forward_pass(self, Z: np.ndarray) -> np.ndarray:
        self.input_prev = Z
        self.output_prev = self.nonlinearity.forward_pass(Z)
        return self.output_prev

    def backward_pass(self, gamma: np.ndarray) -> np.ndarray:
        return self.nonlinearity.backward_pass(gamma, self.input_prev)

class OpticalMeshNetworkLayer(NetworkLayer):
    '''Base class for any network layer consisting of an optical mesh of phase shifters and MZIs'''

    def __init__(self, input_size: int, output_size: int, initializer=None):
        '''
        Initialize the OpticalMeshNetworkLayer
        :param input_size: number of input waveguides
        :param output_size: number of output waveguides
        :param initializer: optional initializer method (WIP)
        '''
        super().__init__(input_size, output_size, initializer=initializer)
        self.mesh: OpticalMesh = None

    def forward_pass(self, X: np.ndarray, cache_fields=False, use_partial_vectors=False) -> np.ndarray:
        raise NotImplementedError('forward_pass() must be overridden in child class!')

    def backward_pass(self, delta: np.ndarray, cache_fields=False, use_partial_vectors=False) -> np.ndarray:
        raise NotImplementedError('backward_pass() must be overridden in child class!')

class ClementsLayer(OpticalMeshNetworkLayer):
    '''Performs a unitary NxM operator with MZIs arranged in a Clements decomposition. If M=N then the layer can
    perform any arbitrary unitary operator
    '''

    def __init__(self, N: int, M=None, include_phase_shifter_layer=True, initializer=None):
        '''
        Initialize the ClementsLayer
        :param N: number of input and output waveguides
        :param M: number of MZI columns; equal to N by default
        :param include_phase_shifter_layer: if true, include a layer of single-mode phase shifters at the beginning of
        the mesh (required to implement arbitrary unitary)
        :param initializer: optional initializer method (WIP)
        '''
        super().__init__(N, N, initializer=initializer)

        layers = []
        if include_phase_shifter_layer:
            layers.append(PhaseShifterLayer(N))

        if M is None:
            M = N

        for layer_index in range(M):
            if N % 2 == 0:  # even number of waveguides
                if layer_index % 2 == 0:
                    layers.append(MZILayer.from_waveguide_indices(N, list(range(0, N))))
                else:
                    layers.append(MZILayer.from_waveguide_indices(N, list(range(1, N - 1))))
            else:  # odd number of waveguides
                if layer_index % 2 == 0:
                    layers.append(MZILayer.from_waveguide_indices(N, list(range(0, N - 1))))
                else:
                    layers.append(MZILayer.from_waveguide_indices(N, list(range(1, N))))

        self.mesh = OpticalMesh(N, layers)

    def forward_pass(self, X: np.ndarray, cache_fields=False, use_partial_vectors=False) -> np.ndarray:
        '''
        Compute the forward pass
        :param X: input electric fields
        :param cache_fields: if true, fields are cached
        :param use_partial_vectors: if true, use partial vector method to speed up transfer matrix computations
        :return: output fields for next ONN layer

        '''
        self.input_prev = X
        if cache_fields:
            self.mesh.forward_fields = self.mesh.compute_phase_shifter_fields(
                X, align="right", use_partial_vectors=use_partial_vectors)
            self.output_prev = np.copy(self.mesh.forward_fields[-1][-1])
        else:
            self.output_prev = np.dot(self.mesh.get_transfer_matrix(), X)

        return self.output_prev

    def backward_pass(self, delta: np.ndarray, cache_fields=False, use_partial_vectors=False) -> np.ndarray:
        '''
        Compute the backward pass
        :param delta: adjoint "output" electric fields backpropagated from the next ONN layer
        :param cache_fields: if true, fields are cached
        :param use_partial_vectors: if true, use partial vector method to speed up transfer matrix computations
        :return: adjoint "input" fields for previous ONN layer
        '''
        if cache_fields:
            self.mesh.adjoint_fields = self.mesh.compute_adjoint_phase_shifter_fields(
                delta, align="right", use_partial_vectors=use_partial_vectors)
            if isinstance(self.mesh.layers[0], PhaseShifterLayer):
                return np.dot(self.mesh.layers[0].get_transfer_matrix().T, self.mesh.adjoint_fields[-1][-1])
            else:
                raise ValueError("Field_store will not work in this case, please set to False")
        else:
            return np.dot(self.mesh.get_transfer_matrix().T, delta)

class ReckLayer(OpticalMeshNetworkLayer):
    '''Performs a unitary NxN operator with MZIs arranged in a Reck decomposition'''

    def __init__(self, N: int, include_phase_shifter_layer=True, initializer=None, phases=[(None, None)], loss=0, phase_uncert=0.0):
        '''
        Initialize the ReckLayer
        :param N: number of input and output waveguides
        :param include_phase_shifter_layer: if true, include a layer of single-mode phase shifters at the beginning of
        the mesh (required to implement arbitrary unitary)
        :param initializer: optional initializer method (WIP)
        '''
        super().__init__(N, N, initializer=initializer)
        self.phase_uncert = phase_uncert
        self.loss = loss
        self.N = N

        layers = []
        if include_phase_shifter_layer:
            layers.append(PhaseShifterLayer(N))

        # Get MZI waveguide limits, upper and lower, for the Reck configuration
        mzi_limits_upper = [i for i in range(1, N)] + [i for i in range(N - 2, 1 - 1, -1)]
        mzi_limits_lower = [(i + 1) % 2 for i in mzi_limits_upper]
        
        if (None, None) in phases:
            phases = [(None, None) for _ in range(int(N*(N-1)/2))]
            # print('None for Theta and Phi')

        mzi_nums = [int(len(range(start, end+1))/2) for start, end in zip(mzi_limits_lower, mzi_limits_upper)] # get the number of MZIs in this component layer

        phases_mzi_layer = []
        idx = 0
        for ii in mzi_nums:
            phases_layer = []
            for jj in range(ii):
                phases_layer.append(phases[idx])
                idx += 1
            phases_mzi_layer.append(phases_layer)

        # create every layer of MZIs within the Reck Mesh
        for start, end, phases in zip(mzi_limits_lower, mzi_limits_upper, phases_mzi_layer):
            thetas = [phase[0] for phase in phases]
            phis = [phase[1] for phase in phases]
            layers.append(MZILayer.from_waveguide_indices(N, list(range(start, end + 1)), loss=loss, thetas=thetas, phis=phis, phase_uncert=self.phase_uncert))

        self.mesh = OpticalMesh(N, layers)

    def set_phases_uncert_loss(self, Phases, phase_uncert, loss):
        # Get MZI waveguide limits, upper and lower, for the Reck configuration
        mzi_limits_upper = [i for i in range(1, self.N)] + [i for i in range(self.N - 2, 1 - 1, -1)]
        mzi_limits_lower = [(i + 1) % 2 for i in mzi_limits_upper]
        mzi_nums = [int(len(range(start, end+1))/2) for start, end in zip(mzi_limits_lower, mzi_limits_upper)] # get the number of MZIs in this component layer
        layers = []
        phases_mzi_layer = []
        idx = 0
        for ii in mzi_nums:
            phases_layer = []
            for jj in range(ii):
                phases_layer.append(Phases[idx])
                idx += 1
            phases_mzi_layer.append(phases_layer)
        # create every layer of MZIs within the Reck Mesh
        for start, end, phases in zip(mzi_limits_lower, mzi_limits_upper, phases_mzi_layer):
            thetas = [phase[0] for phase in phases]
            phis = [phase[1] for phase in phases]
            layers.append(MZILayer.from_waveguide_indices(self.N, list(range(start, end + 1)), loss=loss, thetas=thetas, phis=phis, phase_uncert=phase_uncert))
        self.mesh = OpticalMesh(self.N, layers)


    def forward_pass(self, X: np.ndarray) -> np.ndarray:
        self.input_prev = X
        self.output_prev = np.dot(self.mesh.get_transfer_matrix(), X)
        return self.output_prev

    def backward_pass(self, delta: np.ndarray) -> np.ndarray:
        return np.dot(self.mesh.get_transfer_matrix().T, delta)

class flipped_ReckLayer(OpticalMeshNetworkLayer):
    '''Performs a unitary NxN operator with MZIs arranged in a Reck decomposition, but flipped. This means that the triangle is facing
    up rather than down, like in the originali Reck mesh'''

    def __init__(self, N: int, include_phase_shifter_layer=True, initializer=None, loss=0, phases=[(None, None)],  phase_uncert=0.0):
        '''
        Initialize the ReckLayer
        :param N: number of input and output waveguides
        :param include_phase_shifter_layer: if true, include a layer of single-mode phase shifters at the beginning of
        the mesh (required to implement arbitrary unitary)
        :param initializer: optional initializer method (WIP)
        '''
        super().__init__(N, N, initializer=initializer)
        self.phase_uncert = phase_uncert
        self.loss = loss
        self.N = N
        layers = []

        if include_phase_shifter_layer:
            layers.append(PhaseShifterLayer(N))

        # Get MZI waveguide limits, upper and lower, for the inverse Reck configuration
        mzi_limits_lower = [i for i in range(N - 2, 0, -1)] + [i for i in range(0, N - 1)]
        mzi_limits_upper = [(N - 1) - i % 2 for i in range(len(mzi_limits_lower))]

        if (None, None) in phases:
            phases = [(None, None) for _ in range(int(N*(N-1)/2))]

        mzi_nums = [int(len(range(start, end+1))/2) for start, end in zip(mzi_limits_lower, mzi_limits_upper)] # get the number of MZIs in this component layer

        phases_mzi_layer = []
        idx = 0
        for ii in mzi_nums:
            phases_layer = []
            for jj in range(ii):
                phases_layer.append(phases[idx])
                idx += 1
            phases_mzi_layer.append(phases_layer)

        # create every layer of MZIs within the Reck Mesh
        for start, end, phases in zip(mzi_limits_lower, mzi_limits_upper, phases_mzi_layer):
            thetas = [phase[0] for phase in phases]
            phis = [phase[1] for phase in phases]
            layers.append(MZILayer.from_waveguide_indices(N, list(range(start, end + 1)), loss=loss, thetas=thetas, phis=phis, phase_uncert=self.phase_uncert))

        self.mesh = OpticalMesh(N, layers)
    

    def set_phases_uncert_loss(self, Phases, phase_uncert, loss):
        # Essentially recreate the network with specified phases
        # Get MZI waveguide limits, upper and lower, for the inverse Reck configuration
        mzi_limits_lower = [i for i in range(self.N - 2, 0, -1)] + [i for i in range(0, self.N - 1)]
        mzi_limits_upper = [(self.N - 1) - i % 2 for i in range(len(mzi_limits_lower))]
        mzi_nums = [int(len(range(start, end+1))/2) for start, end in zip(mzi_limits_lower, mzi_limits_upper)] # get the number of MZIs in this component layer
        phases_mzi_layer = []
        layers = []
        idx = 0
        for ii in mzi_nums:
            phases_layer = []
            for jj in range(ii):
                phases_layer.append(Phases[idx])
                idx += 1
            phases_mzi_layer.append(phases_layer)
        # create every layer of MZIs within the Reck Mesh
        for start, end, phases in zip(mzi_limits_lower, mzi_limits_upper, phases_mzi_layer):
            thetas = [phase[0] for phase in phases]
            phis = [phase[1] for phase in phases]
            layers.append(MZILayer.from_waveguide_indices(self.N, list(range(start, end + 1)),  thetas=thetas, phis=phis, phase_uncert=phase_uncert, loss=loss))
        self.mesh = OpticalMesh(self.N, layers)


    def forward_pass(self, X: np.ndarray) -> np.ndarray:
        self.input_prev = X
        self.output_prev = np.dot(self.mesh.get_transfer_matrix(), X)
        return self.output_prev

    def backward_pass(self, delta: np.ndarray) -> np.ndarray:
        return np.dot(self.mesh.get_transfer_matrix().T, delta)

class DMM_layer(OpticalMeshNetworkLayer):
    def __init__(self, N: int, initializer=None, loss=0, phases=[(None, None)], phase_uncert=0.0):
        '''
        Initialize DMM (diagonal MZI layer to create the Sigma portion of the SVD Decomposition)
        :param N: Number of MZIs in the Mesh, will be twice as many as the Reck and Reck' layer.
        thetas anbd phis will be equal to N/2 in this layer, where N = twice the number of waveguides coming
        from the Reck layer
        '''
        super().__init__(N, N, initializer=initializer)
        self.phase_uncert = phase_uncert
        self.loss = loss
        self.N = N

        layers = []
        # Get MZI waveguide limits, upper and lower, for the DMM configuration
        mzi_limits_upper = [N - 1]
        mzi_limits_lower = [0]
        if (None, None) in phases:
            phases = [(None, None) for _ in range(int(N))]
        mzi_nums = [int(len(range(start, end+1))/2) for start, end in zip(mzi_limits_lower, mzi_limits_upper)] # get the number of MZIs in this component layer

        phases_mzi_layer = []
        idx = 0
        for ii in mzi_nums:
            phases_layer = []
            for jj in range(ii):
                phases_layer.append(phases[idx])
                idx += 1
            phases_mzi_layer.append(phases_layer)

        # now separate the phases using the number of MZIs in their respective component layer
        for start, end, phases in zip(mzi_limits_lower, mzi_limits_upper, phases_mzi_layer):
            thetas = [phase[0] for phase in phases]
            phis = [phase[1] for phase in phases]
            layers.append(MZILayer.from_waveguide_indices(N, list(range(start, end + 1)), loss=loss, thetas=thetas, phis=phis, phase_uncert=self.phase_uncert))

        self.mesh = OpticalMesh(N, layers)

    def set_phases_uncert_loss(self, Phases, phase_uncert, loss):
        layers = []
        # Get MZI waveguide limits, upper and lower, for the DMM configuration
        mzi_limits_upper = [self.N - 1]
        mzi_limits_lower = [0]
        mzi_nums = [int(len(range(start, end+1))/2) for start, end in zip(mzi_limits_lower, mzi_limits_upper)] # get the number of MZIs in this component layer
        phases_mzi_layer = []
        idx = 0
        for ii in mzi_nums:
            phases_layer = []
            for jj in range(ii):
                phases_layer.append(Phases[idx])
                idx += 1
            phases_mzi_layer.append(phases_layer)
        # now separate the phases using the number of MZIs in their respective component layer
        for start, end, phases in zip(mzi_limits_lower, mzi_limits_upper, phases_mzi_layer):
            thetas = [phase[0] for phase in phases]
            phis = [phase[1] for phase in phases]
            layers.append(MZILayer.from_waveguide_indices(self.N, list(range(start, end + 1)),  thetas=thetas, phis=phis, phase_uncert=phase_uncert, loss=loss))
        self.mesh = OpticalMesh(self.N, layers)


    def forward_pass(self, X: np.ndarray, pKeep=0.8) -> np.ndarray:
        self.input_prev = X
        self.output_prev = np.dot(self.mesh.get_transfer_matrix(), X)
        return self.output_prev

    def backward_pass(self, delta: np.ndarray) -> np.ndarray:
        return np.dot(self.mesh.get_transfer_matrix().T, delta)

class ReckLayer_H(OpticalMeshNetworkLayer): # Hermitian Transpose of a Reck Layer
    '''Performs a unitary NxN operator with MZIs arranged in a Reck decomposition,
    but in the inverse setup (Reck^-1), to create the V^\dagger
    portion of the SVD Decomposition'''

    def __init__(self, N: int, include_phase_shifter_layer=True, initializer=None, loss=0, thetas=[None], phis=[None]):
        '''
        Initialize the ReckLayer
        :param N: number of input and output waveguides
        :param include_phase_shifter_layer: if true, include a layer of single-mode phase shifters at the beginning of
        the mesh (required to implement arbitrary unitary)
        :param initializer: optional initializer method (WIP)
        '''
        super().__init__(N, N, initializer=initializer)

        layers = []
        if include_phase_shifter_layer:
            layers.append(PhaseShifterLayer(N))

        # get the wabeguide limits of the hermitian transpose of the Reck layer (will be the flipped Reck layer)
        mzi_limits_lower = [i for i in range(N - 2, 0, -1)] + [i for i in range(0, N - 1)]
        mzi_limits_upper = [(N - 1) - i % 2 for i in range(len(mzi_limits_lower))]

        # Assign Thetas and phis
        # if theta/phis is [None], make it as long as the number of MZIs in Reck layer
        if None in thetas:
            self.thetas = [None]*int(N*(N-1)/2)
        else:
            assert len(thetas) == int(N*(N-1)/2) # Otherwise, check if Number of phases is equal to number of MZIs
            self.thetas = thetas

        if None in phis:
            self.phis = [None]*int(N*(N-1)/2)
        else:
            assert len(phis) == int(N*(N-1)/2)
            self.phis = phis

        # Combine the phases into (theta, phi) tuples
        phases = [(theta, phi) for (theta, phi) in zip(self.thetas, self.phis)]
        # Get number of mzi in each separate component layer
        mzi_nums = [int(len(range(start, end+1))/2) for start, end in zip(mzi_limits_lower, mzi_limits_upper)]

        # now separate the (theta, phi) tuples into their respective component layer
        phases_mzi_layer = []
        idx = 0
        for ii in mzi_nums:
            phases_layer = []
            for jj in range(ii):
                phases_layer.append(phases[idx])
                idx += 1
            phases_mzi_layer.append(phases_layer)

        # print(mzi_limits_lower)
        # print(mzi_limits_upper)
        # Finally, create each MZI Layer (the hermitian transpose of the Reck layer's MZILayer)
        for start, end, phases in zip(mzi_limits_lower, mzi_limits_upper, phases_mzi_layer):
            thetas = [phase[0] for phase in phases]
            phis = [phase[1] for phase in phases]
            layers.append(MZILayer_H.from_waveguide_indices(N, list(range(start, end + 1)), loss=loss, thetas=thetas, phis=phis))

        self.mesh = OpticalMesh(N, layers)

    def forward_pass(self, X: np.ndarray) -> np.ndarray:
        self.input_prev = X
        self.output_prev = np.dot(self.mesh.get_transfer_matrix(), X)
        return self.output_prev

    def backward_pass(self, delta: np.ndarray) -> np.ndarray:
        return np.dot(self.mesh.get_transfer_matrix().T, delta)
