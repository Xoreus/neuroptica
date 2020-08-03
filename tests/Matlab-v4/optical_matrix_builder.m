function [U, D] = optical_matrix_builder(Theta, Phi)
% optical_matrix_builder
% This code uses the general optical
% matrix equation, U = D6*D5*D4*D3*D2*D1.
%
% Inputs: thetas - 1 by 6 array of thetas, controlling the power splitting
%                  ratio of the MZIs, Theta = [0, pi]. 
%         phis - 1 by 6 array of phis, controlling the outgoing phase of
%                the upper output arm of every MZI. Phi = [0, 2*pi]
% Outputs: U - a unitary matrix based on the channel mixing by the MZIs 1-6
%         
%
% Author: Simon Geoffroy-Gagnon, addapted from code by Farhad Shokraneh
% Edited: 2019/03/23
%%

% From 0 to pi --- 0 goes at bar, pi goes at cross
theta1 = Theta(1);   % top corner 
theta2 = Theta(2);  % middle
theta3 = Theta(3);  % top corner
theta4 = Theta(4); % bottom corner
theta5 = Theta(5); % middle
theta6 = Theta(6); % top corner

%%%%%%%%%% External Phase shifter after the second DC of the MZIs
phi1 = Phi(1);   % From 0 to 2pi
phi2 = Phi(2);
phi3 = Phi(3);    
phi4 = Phi(4); 
phi5 = Phi(5);    
phi6 = Phi(6);  


%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%% D1 %%%%%%%%%%%%%%%%%%%%%
D1 = exp(1j*(theta1+pi)/2)*[exp(1j*phi1), 0; 0, 1]*...
    [sin(theta1/2), cos(theta1/2); cos(theta1/2), -sin(theta1/2)];
D1 = [D1(1,1) D1(1,2)  0 0;
      D1(2,1), D1(2,2) 0 0;
      0        0       1 0;
      0        0       0 1];
%%%%%%%%%%%%%%%% D2 %%%%%%%%%%%%%%%%%%%
D2 = exp(1j*(theta2+pi)/2)*[exp(1j*phi2), 0; 0, 1]*...
    [sin(theta2/2), cos(theta2/2); cos(theta2/2), -sin(theta2/2)];
D2 = [1  0        0       0;
      0  D2(1,1)  D2(1,2) 0;
      0  D2(2,1)  D2(2,2) 0;
      0  0        0       1];

%%%%%%%%%%%%%%% WHY is D3 at the top corner? %%%%%%%%%%%%%%%%%%%%%%%
D3 = exp(1j*(theta3+pi)/2)*[exp(1j*phi3), 0; 0, 1]*...
    [sin(theta3/2), cos(theta3/2); cos(theta3/2), -sin(theta3/2)];
D3 = [D3(1,1) D3(1,2)  0 0;
      D3(2,1), D3(2,2) 0 0;
      0        0       1 0;
      0        0       0 1];

%%%%%%%%%%%%%%%%%% D4 %%%%%%%%%%%%%%%%%
D4 = exp(1j*(theta4+pi)/2)*[exp(1j*phi4), 0; 0, 1]*...
    [sin(theta4/2), cos(theta4/2); cos(theta4/2), -sin(theta4/2)];
D4 = [1 0 0        0;
      0 1 0        0;
      0 0 D4(1,1), D4(1,2);
      0 0 D4(2,1), D4(2,2)];

%%%%%%%%%%%%%%%%% D5 %%%%%%%%%%%%%%%%%%
D5 = exp(1j*(theta5+pi)/2)*[exp(1j*phi5), 0; 0, 1]*...
    [sin(theta5/2), cos(theta5/2); cos(theta5/2), -sin(theta5/2)];
D5 = [1  0        0       0;
      0  D5(1,1)  D5(1,2) 0;
      0  D5(2,1)  D5(2,2) 0;
      0  0        0       1];

%%%%%%%%%%%%%%%% D6 %%%%%%%%%%%%%%%%%%%
D6 = exp(1j*(theta6+pi)/2)*[exp(1j*phi6), 0; 0, 1]*...
    [sin(theta6/2), cos(theta6/2); cos(theta6/2), -sin(theta6/2)];
D6 = [D6(1,1) D6(1,2)  0 0;
      D6(2,1), D6(2,2) 0 0;
      0        0       1 0;
      0        0       0 1];



% 4x4 matrix
U = D6*D5*D4*D3*D2*D1;

% ONN Weight Matrix
D = U;
end
