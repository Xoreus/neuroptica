import numpy as np


def MZI(theta, phi):
    return 1j*np.exp(1j*theta/2)*np.matrix([[np.exp(1j*phi)*np.sin(theta/2), np.exp(1j*phi)*np.cos(theta/2)],
                                            [np.cos(theta/2),                -np.sin(theta/2)]])



theta = [0, 1, 1, 0, 0, 0]
phi = [0, 1, 1, 0, 0, 0]

D0 = MZI(theta[0],phi[0])
D0 = np.matrix([[D0[0,0],D0[0,1], 0, 0],
      [D0[1,0],D0[1,1], 0, 0],
      [0, 0, 1, 0],
      [0, 0, 0, 1]])

D1 = MZI(theta[1],phi[1])
D1 = np.matrix([[1, 0, 0, 0],
      [0, D1[0,0],D1[0,1], 0,],
      [0, D1[1,0],D1[1,1],0],
      [0, 0, 0, 1]])

D2 = MZI(theta[2],phi[2])
D2 = np.matrix([[D2[0,0],D2[0,1], 0, 0],
      [D2[1,0],D2[1,1], 0, 0],
      [0, 0, 1, 0],
      [0, 0, 0, 1]])

D3 = MZI(theta[3],phi[3])
D3 = np.matrix([[1, 0, 0, 0],
      [0, 1, 0, 0],
      [0, 0, D3[0,0], D3[0, 1]],
      [0, 0, D3[1,0], D3[1, 1]]])

D4 = MZI(theta[4],phi[4])
D4 = np.matrix([[1, 0, 0, 0],
      [0, D4[0,0],D4[0,1], 0,],
      [0, D4[1,0],D4[1,1],0],
      [0, 0, 0, 1]])

D5 = MZI(theta[5],phi[5])
D5 = np.matrix([[D5[0,0],D5[0,1], 0, 0],
      [D5[1,0],D5[1,1], 0, 0],
      [0, 0, 1, 0],
      [0, 0, 0, 1]])

D = D5*D4*D3*D2*D1*D0
np.set_printoptions(precision=3)
print(D)
