import numpy as np
import sys
sys.path.append('../')
import neuroptica as neu
import matplotlib.pyplot as plt


eo_settings = { 'alpha': 0.5,
                'g':     5 * np.pi,
                'phi_b': 1 * np.pi }

eo_activation = neu.ElectroOpticActivation(1, **eo_settings)

x = np.linspace(0, 1, 100)
plt.plot(x, np.real(eo_activation.forward_pass(x)),label="Re")
plt.plot(x, np.imag(eo_activation.forward_pass(x)),label="Im")
plt.plot(x, np.abs(eo_activation.forward_pass(x)), label="Abs")
plt.xlabel("Input field (a.u.)")
plt.ylabel("Output field (a.u.)")
plt.legend()
plt.show()


