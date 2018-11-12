from functools import reduce
from typing import Callable, Dict, Iterable, List, Type

import numpy as np

from neuroptica.components.components import MZI, OpticalComponent, PhaseShifter
from neuroptica.settings import NP_COMPLEX


# component_type = deferred_type()
# @jitclass(OrderedDict({
#     "N": int32,
#     "components":component_type[:]
# }))
class ComponentLayer:
    '''
    Base class for a physical column of optical components
    '''

    def __init__(self, N: int, components: List[Type[OpticalComponent]]):
        self.N = N
        self.components = components

    def __iter__(self) -> Iterable[Type[OpticalComponent]]:
        yield from self.components

    def all_tunable_params(self) -> Iterable[float]:
        raise NotImplementedError

    def get_transfer_matrix(self) -> np.ndarray:
        raise NotImplementedError("get_transfer_matrix() must be extended for child classes!")


# mzi_type = deferred_type().define(MZI.class_type.instance_type)
# @jitclass(OrderedDict({
#     "mzis": mzi_type[:]
# }))
class MZILayer(ComponentLayer):
    '''
    Represents a physical column of MZI's attached to an ensemble of waveguides
    '''

    def __init__(self, N: int, mzis: List[MZI]):
        super().__init__(N, mzis)
        self.mzis = mzis

    def __iter__(self) -> Iterable[MZI]:
        yield from self.mzis

    def all_tunable_params(self):
        for mzi in self.mzis:
            yield mzi.theta
            yield mzi.phi

    @classmethod
    def from_waveguide_indices(cls, N: int, waveguide_indices: List[int]):
        '''Create an MZI layer from a list of an even number of input/output indices. Each pair of waveguides in the
        iteration order will be assigned to an MZI'''
        assert len(waveguide_indices) % 2 == 0 and len(waveguide_indices) <= N and \
               len(np.unique(waveguide_indices)) == len(waveguide_indices), \
            "Waveguide must have an even number <= N of indices which are all unique"
        mzis = []
        for i in range(0, len(waveguide_indices), 2):
            mzis.append(MZI(waveguide_indices[i], waveguide_indices[i + 1]))
        return cls(N, mzis)

    @staticmethod
    def verify_inputs(N: int, mzis: List[MZI]):
        '''Checks that the input MZIs are valid'''
        assert len(mzis) <= N // 2, "Too many MZIs for layer with {} waveguides".format(N)
        input_ports = np.array([[mzi.m, mzi.n] for mzi in mzis]).flatten()
        assert len(np.unique(input_ports)) == len(input_ports), "MZIs share duplicate input ports!"

    def get_transfer_matrix(self, add_uncertainties=False) -> np.ndarray:
        T = np.eye(self.N, dtype=NP_COMPLEX)
        for mzi in self.mzis:
            U = mzi.get_transfer_matrix(add_uncertainties)
            m, n = mzi.m, mzi.n
            T[m][m] = U[0, 0]
            T[m][n] = U[0, 1]
            T[n][m] = U[1, 0]
            T[n][n] = U[1, 1]
        return T

    def get_partial_transfer_matrices(self, backward=False, cumulative=True,
                                      add_uncertainties=False) -> List[np.ndarray]:
        '''Return a list of 4 partial transfer matrices for the entire MZI layer corresponding to (1) after first BS in
        each MZI, (2) after theta shifter, (3) after second BS, and (4) after phi shifter. Order is reversed in the
        backwards case'''

        Ttotal = np.eye(self.N, dtype=NP_COMPLEX)

        partial_transfer_matrices = []
        # Compute the (non-cumulative) partial transfer matrices for each MZI
        all_mzi_partials = [mzi.get_partial_transfer_matrices(backward=backward, cumulative=False,
                                                              add_uncertainties=add_uncertainties) for mzi in self.mzis]

        for depth in range(len(all_mzi_partials[0])):
            # Iterate over each sub-component at a given depth
            T = np.eye(self.N, dtype=NP_COMPLEX)

            for i, mzi in enumerate(self.mzis):
                U = all_mzi_partials[i][depth]
                m, n = mzi.m, mzi.n
                T[m][m] = U[0, 0]
                T[m][n] = U[0, 1]
                T[n][m] = U[1, 0]
                T[n][n] = U[1, 1]

            if cumulative:
                Ttotal = np.dot(T, Ttotal)
                partial_transfer_matrices.append(Ttotal)
            else:
                partial_transfer_matrices.append(T)

        return partial_transfer_matrices

    # @staticmethod
    # @njit
    # def _get_partial_transfer_matrices(mzis:List[MZI], N, T_base,
    #                                    backward=False, cumulative=True, add_uncertainties=False) -> List[np.ndarray]:
    #
    #     Ttotal = np.copy(T_base) #np.eye(N, dtype=NP_COMPLEX)
    #
    #     partial_transfer_matrices = []
    #
    #     # Compute the (non-cumulative) partial transfer matrices for each MZI
    #     all_mzi_partials = [mzi.get_partial_transfer_matrices(backward=backward, cumulative=False,
    #                                                           add_uncertainties=add_uncertainties) for mzi in mzis]
    #
    #     for depth in range(len(all_mzi_partials[0])):
    #         # Iterate over each sub-component at a given depth
    #         # T = np.eye(N, dtype=NP_COMPLEX)
    #         T= np.copy(T_base)
    #
    #         for i, mzi in enumerate(mzis):
    #             U = all_mzi_partials[i][depth]
    #             m, n = mzi.m, mzi.n
    #             T[m][m] = U[0, 0]
    #             T[m][n] = U[0, 1]
    #             T[n][m] = U[1, 0]
    #             T[n][n] = U[1, 1]
    #
    #         if cumulative:
    #             Ttotal = np.dot(T, Ttotal)
    #             partial_transfer_matrices.append(Ttotal)
    #         else:
    #             partial_transfer_matrices.append(T)
    #
    #     return partial_transfer_matrices


# phase_shifter_type = deferred_type().define(PhaseShifter.class_type.instance_type)
# @jitclass(OrderedDict({
#     "phase_shifters": phase_shifter_type[:]
# }))
class PhaseShifterLayer(ComponentLayer):
    '''
    Represents a column of N single-mode phase shifters
    '''

    def __init__(self, N: int, phase_shifters: List[PhaseShifter] = None):
        super().__init__(N, phase_shifters)

        if phase_shifters is None:
            phase_shifters = [PhaseShifter(m) for m in range(N)]
        self.phase_shifters = phase_shifters

    def __iter__(self) -> Iterable[PhaseShifter]:
        yield from self.phase_shifters

    def all_tunable_params(self):
        for phase_shifter in self.phase_shifters:
            yield phase_shifter.phi

    def get_transfer_matrix(self, add_uncertainties=False) -> np.ndarray:
        T = np.eye(self.N, dtype=NP_COMPLEX)
        for phase_shifter in self.phase_shifters:
            m = phase_shifter.m
            T[m][m] = phase_shifter.get_transfer_matrix()[0, 0]
        return T


class OpticalMesh:
    '''
    Represents a mesh consisting of several layers of optical components
    '''

    def __init__(self, N: int, layers: List[Type[ComponentLayer]]):
        self.N = N
        self.layers = layers

    def __iter__(self) -> Iterable[ComponentLayer]:
        yield from self.layers

    def all_tunable_params(self) -> Iterable[float]:
        for layer in self.layers:
            for param in layer.all_tunable_params():
                yield param

    def all_tunable_components(self) -> Iterable[Type[OpticalComponent]]:
        for layer in self.layers:
            yield from layer

    @staticmethod
    def verify_inputs(N: int, layers: List[Type[ComponentLayer]]):
        assert all([N == layer.N for layer in layers]), "Dimension mismatch in layers!"

    def get_transfer_matrix(self) -> np.ndarray:
        return reduce(np.dot, [layer.get_transfer_matrix() for layer in reversed(self.layers)])
        # return self._get_transfer_matrix([layer.get_transfer_matrix() for layer in self.layers])

    # @staticmethod
    # @njit(parallel=True)
    # def _get_transfer_matrix(transfer_matrices: List[np.ndarray]) -> np.ndarray:
    #     T = transfer_matrices[0]
    #     for transfer_matrix in transfer_matrices[1:]:
    #         T = np.dot(transfer_matrix, T)
    #     return T

    # TODO
    def get_partial_transfer_matrices(self, backward=False, cumulative=True) -> List[np.ndarray]:
        '''Return the cumulative transfer matrices following each layer in the mesh'''
        partial_transfer_matrices = []
        Ttotal = np.eye(self.N, dtype=NP_COMPLEX)
        layers = reversed(self.layers) if backward else self.layers
        for layer in layers:
            T = layer.get_transfer_matrix()
            if backward:
                T = T.T
            Ttotal = np.dot(T, Ttotal)  # needs to be (T . Ttotal), left multiply
            partial_transfer_matrices.append(Ttotal)
        return partial_transfer_matrices

    def compute_phase_shifter_fields(self, X: np.ndarray, align="right") -> List[List[np.ndarray]]:
        '''
        Compute the foward-pass field at the left/right of each phase shifter in the mesh
        :param X: input field to the mesh
        :return: a list of (list of field values to the left/right of each phase shifter in a layer) for each layer
        '''

        fields = []

        X_current = np.copy(X)

        for layer in self.layers:

            if isinstance(layer, MZILayer):
                partial_transfer_matrices = layer.get_partial_transfer_matrices(backward=False, cumulative=True)
                bs1_T, theta_T, bs2_T, phi_T = partial_transfer_matrices
                if align == "right":
                    fields.append([np.dot(theta_T, X_current), np.dot(phi_T, X_current)])
                elif align == "left":
                    fields.append([np.dot(bs1_T, X_current), np.dot(bs2_T, X_current)])
                else:
                    raise ValueError('align must be "left" or "right"!')

            elif isinstance(layer, PhaseShifterLayer):
                if align == "right":
                    fields.append([np.dot(layer.get_transfer_matrix(), X_current)])
                elif align == "left":
                    fields.append([np.copy(X_current)])
                else:
                    raise ValueError('align must be "left" or "right"!')

            else:
                raise TypeError("Layer is not instance of MZILayer or PhaseShifterLayer!")

            X_current = np.dot(layer.get_transfer_matrix(), X_current)

        return fields

    def compute_adjoint_phase_shifter_fields(self, delta: np.ndarray, align="right") -> List[List[np.ndarray]]:
        '''
        Compute the backward-pass field at the left/right of each phase shifter in the mesh
        :param delta: input adjoint field to the mesh
        :return: a list of (list of field values to the left/right of each phase shifter in a layer) for each layer
        The ordering of the list is the opposite as in compute_phase_shifter_fields()
        '''

        adjoint_fields = []

        delta_current = np.copy(delta)

        for layer in reversed(self.layers):

            if isinstance(layer, MZILayer):
                partial_transfer_matrices_inv = layer.get_partial_transfer_matrices(backward=True, cumulative=True)
                phi_T_inv, bs2_T_inv, theta_T_inv, bs1_T_inv = partial_transfer_matrices_inv

                if align == "right":
                    adjoint_fields.append([np.copy(delta_current), np.dot(bs2_T_inv, delta_current)])
                elif align == "left":
                    adjoint_fields.append([np.dot(phi_T_inv, delta_current), np.dot(theta_T_inv, delta_current)])
                else:
                    raise ValueError('align must be "left" or "right"!')

            elif isinstance(layer, PhaseShifterLayer):
                if align == "right":
                    adjoint_fields.append([np.copy(delta_current)])
                elif align == "left":
                    adjoint_fields.append([np.dot(layer.get_transfer_matrix().T, delta_current)])
                else:
                    raise ValueError('align must be "left" or "right"!')

            else:
                raise TypeError("Layer is not instance of MZILayer or PhaseShifterLayer!")

            delta_current = np.dot(layer.get_transfer_matrix().T, delta_current)

        return adjoint_fields

    def adjoint_optimize(self, forward_field: np.ndarray, adjoint_field: np.ndarray,
                         update_fn: Callable,  # update function takes a float and possibly other args and returns float
                         accumulator: Callable[[np.ndarray], float] = np.mean,
                         dry_run=False):

        forward_fields = self.compute_phase_shifter_fields(forward_field, align="right")
        adjoint_fields = self.compute_adjoint_phase_shifter_fields(adjoint_field, align="right")

        gradient_dict = {}

        for layer, layer_fields, layer_fields_adj in zip(self.layers, forward_fields, reversed(adjoint_fields)):

            if isinstance(layer, PhaseShifterLayer):
                A_phi, A_phi_adj = layer_fields[0], layer_fields_adj[0]
                dL_dphi = -1 * np.imag(A_phi * A_phi_adj)
                if dry_run:
                    gradient_dict[layer] = [dL_dphi]

                else:
                    for phase_shifter in layer.phase_shifters:
                        delta_phi = accumulator(dL_dphi[phase_shifter.m])
                        phase_shifter.phi += update_fn(delta_phi)

            elif isinstance(layer, MZILayer):
                A_theta, A_phi = layer_fields
                A_theta_adj, A_phi_adj = reversed(layer_fields_adj)
                dL_dtheta = -1 * np.imag(A_theta * A_theta_adj)
                dL_dphi = -1 * np.imag(A_phi * A_phi_adj)
                if dry_run:
                    gradient_dict[layer] = [dL_dtheta, dL_dphi]

                else:
                    for mzi in layer.mzis:
                        delta_theta = accumulator(dL_dtheta[mzi.m])
                        delta_phi = accumulator(dL_dphi[mzi.m])
                        mzi.theta += update_fn(delta_theta)
                        mzi.phi += update_fn(delta_phi)

            else:
                raise ValueError("Tunable component layer must be phase-shifting!")

        if dry_run:
            return gradient_dict

    def compute_gradients(self, forward_field: np.ndarray, adjoint_field: np.ndarray) \
            -> Dict[Type[OpticalComponent], np.ndarray]:

        forward_fields = self.compute_phase_shifter_fields(forward_field, align="right")
        adjoint_fields = self.compute_adjoint_phase_shifter_fields(adjoint_field, align="right")

        gradients = {}

        for layer, layer_fields, layer_fields_adj in zip(self.layers, forward_fields, reversed(adjoint_fields)):

            if isinstance(layer, PhaseShifterLayer):
                A_phi, A_phi_adj = layer_fields[0], layer_fields_adj[0]
                dL_dphi = -1 * np.imag(A_phi * A_phi_adj)
                for phase_shifter in layer.phase_shifters:
                    gradients[phase_shifter] = np.array([dL_dphi[phase_shifter.m]])

            elif isinstance(layer, MZILayer):
                A_theta, A_phi = layer_fields
                A_theta_adj, A_phi_adj = reversed(layer_fields_adj)
                dL_dtheta = -1 * np.imag(A_theta * A_theta_adj)
                dL_dphi = -1 * np.imag(A_phi * A_phi_adj)
                for mzi in layer.mzis:
                    gradients[mzi] = np.array([dL_dtheta[mzi.m], dL_dphi[mzi.m]])

            else:
                raise ValueError("Tunable component layer must be phase-shifting!")

        return gradients