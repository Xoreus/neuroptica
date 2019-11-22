import sys
sys.path.append('/home/simon/Documents/neuroptica')
import neuroptica as neu

def ONN_creation(layers='R_D_I_P', N=4, loss=0, phase_uncert=0, Nonlinearity=neu.Sigmoid(4), Phases=[(None, None)]):

    layers = layers.replace('_', '') 
    layers = ''.join(char if char != 'D' else 'AMD' for char in layers) # D really means AddMask, DMM, DropMask
    if layers[-1] != 'P': # Add a PD nonlin as the last output if its not there
        layers = layers + 'P'

    layer_dict = {
            'R':neu.ReckLayer(N, include_phase_shifter_layer=False, loss=loss, phase_uncert=phase_uncert),
            'I':neu.flipped_ReckLayer(N, include_phase_shifter_layer=False, loss=loss,phase_uncert=phase_uncert), 
            'A':neu.AddMask(2*N), 'M':neu.DMM_layer(2*N, loss=loss, phase_uncert=phase_uncert),
            'D':neu.DropMask(N=2*N, keep_ports=range(0, 2*N, 2)), 
            'N':neu.Activation(Nonlinearity), 
            'P':neu.Activation(neu.AbsSquared(N)),
            'B':neu.Activation(neu.Abs(N))}

    Model = neu.Sequential([layer_dict[layer] for layer in layers])
    return Model

if __name__ == '__main__':
    Model = ONN_creation()
    Phases = Model.get_all_phases()

    Model = ONN_creation()
    Model.set_all_phases_uncerts_losses(Phases)
