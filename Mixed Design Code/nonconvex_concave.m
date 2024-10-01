%% Formulation
clear all
clc 
close all

% rng(77);

% Define Model Parameters (LTI)
N = 2; % length of horizon
m = 1;  % dimension of state
d = 1;  % dimension of control input u
n = 1;  % dimension of the disturbance input w
l = 1;  % dimension of the controlled output z

% A = [1 0 -5;-1 1 0;0 0 1];
% B = [1 -10 0;0 3 1;-1 0 2];
% D = 0.5*eye(3);
% Q = [3 -1 0;
%      -1 2 -1;
%      0 -1 1]; % Q, R has to be symmetric and Q>=0, R>0
% Ru = [2 1 1;
%       1 3 -1;
%       1 -1 3];
% Rw = 7.2254256504167845*eye(3);
A = 2;
B = 1;
D = 1;
Q = 1;
Ru = 1;
Rw = 5;
% initialize a K
% for i = 1:100000000
%     i
% K = 2*(rand(3)-0.5)
% K = [-0.8750    1.2500   -2.5000;
%    -0.1875    0.1250    0.2500;
%    -0.4375    0.6250   -0.7500];
eps = 0;
K = 2 - eps;
% K1 = [-0.8750    1.25   -2.50;
%    -0.1875    0.125    0.25;
%    -0.4375    0.625   -0.75];
% K2 = [-0.8786    1.2407   -2.4715;
%    -0.1878    0.1237    0.2548;
%    -0.4439    0.5820   -0.7212];
% K3 = (K1+K2)/2;
% K1 = [1.44    0.31   -1.18;
%     0.03   -0.13   -0.39;
%     0.36   -1.71    0.24];
% K2 = [-0.08   -0.16   -1.96;
%    -0.13   -1.12    1.28;
%     1.67   -0.91    1.71];
% K1 = [-0.12 0.03 0.63; -0.21 0.14 0.15; -0.06 0.05 0.42];
% K2 = [-0.14    0.14    0.63;
%    -0.215    0.09    0.16;
%    -0.11   -0.01    0.41];
% K3 = (K1+K2)/2;
% L1 = [-0.86 0.97 0.14;
%   -0.82 0.36 0.51;
%    0.98 0.08 -0.20];
% L2 = [-0.70   -0.37    0.09;
%    -0.54   -0.28    0.23;
%     0.74    0.62   -0.51];
% L3 = (L1+L2)/2; 
% L = zeros(3);
% Initialize Compact Matrices
bX = zeros((N+1)*m, 1);
bU = zeros(N*d, 1);
bA = zeros((N+1)*m, (N+1)*m);
bB = zeros((N+1)*m, N*d);
bD = zeros((N+1)*m, N*n);
bQ = zeros((N+1)*m, (N+1)*m);
bRu = zeros(N*d, N*d);
bRw = zeros(N*n, N*n);
bK = zeros(N*d, (N+1)*m);
bK1 = zeros(N*d, (N+1)*m);
bK2 = zeros(N*d, (N+1)*m);
bK3 = zeros(N*d, (N+1)*m);
bL = zeros(N*n, (N+1)*m);
bL1 = zeros(N*n, (N+1)*m);
bL2 = zeros(N*n, (N+1)*m);
bL3 = zeros(N*n, (N+1)*m);
bP = zeros((N+1)*m, (N+1)*m);
bKmask = zeros(N*d, (N+1)*m);
bLmask = zeros(N*n, (N+1)*m);

% Fill Compact Matrices
for i = 0:N
    bQ(i*m+1:(i+1)*m, i*m+1:(i+1)*m) = Q;
    if i~=N
      bA((i+1)*m+1:(i+2)*m, i*m+1:(i+1)*m) = A;
      bB((i+1)*m+1:(i+2)*m, i*d+1:(i+1)*d) = B;
      bD((i+1)*m+1:(i+2)*m, i*n+1:(i+1)*n) = D;
      bRu(i*d+1:(i+1)*d, i*d+1:(i+1)*d) = Ru;
      bRw(i*n+1:(i+1)*n, i*n+1:(i+1)*n) = Rw;
      bK(i*d+1:(i+1)*d, i*m+1:(i+1)*m) = K;
%       bK1(i*d+1:(i+1)*d, i*m+1:(i+1)*m) = K1; 
%       bK2(i*d+1:(i+1)*d, i*m+1:(i+1)*m) = K2; 
%       bK3(i*d+1:(i+1)*d, i*m+1:(i+1)*m) = K3; 
%       bL1(i*n+1:(i+1)*n, i*m+1:(i+1)*m) = L1;
%       bL2(i*n+1:(i+1)*n, i*m+1:(i+1)*m) = L2;
%       bL3(i*n+1:(i+1)*n, i*m+1:(i+1)*m) = L3;
%       bL(i*n+1:(i+1)*n, i*m+1:(i+1)*m) = L;
      bKmask(i*d+1:(i+1)*d, i*m+1:(i+1)*m) = ones(d, m);
      bLmask(i*n+1:(i+1)*n, i*m+1:(i+1)*m) = ones(n, m);
    end
end
% bD = [0 0; 0 0; 0 1];
% bK = [0 0 0; 
%       0 0.5897 0];
% bPK = bQ + bK'*bRu*bK
% for i = 1:N
%     bPK = bQ + bK'*bRu*bK + (bA-bB*bK)'*(bPK + bPK*bD*pinv(bRw-bD'*bPK*bD)*bD'*bPK)*(bA-bB*bK)
% end
% [bPK,KK,LL, info] = idare(bA-bB*bK, bD, bQ+bK'*bRu*bK, -bRw, [], []);
my_bPK = Solve_Riccati_K(N, bA, bB, bD, bQ, bRu, bRw, bK); % for a fixed K, solve the Riccati of opt wrt L
% bL_K = (-bRw + bD'*bPK*bD)\(bD'*bPK*(bA-bB*bK));
% bPK1 = Solve_Riccati_K(N, bA, bB, bD, bQ, bRu, bRw, bK1);
% bPK2 = Solve_Riccati_K(N, bA, bB, bD, bQ, bRu, bRw, bK2);
% bPK3 = Solve_Riccati_K(N, bA, bB, bD, bQ, bRu, bRw, bK3);
% cK1 = min(eig(bRw - bD'*bPK1*bD))
% cK2 = min(eig(bRw - bD'*bPK2*bD))
% cK3 = min(eig(bRw - bD'*bPK3*bD))
% bL_K1 = (-bRw + bD'*bPK1*bD)\(bD'*bPK1*(bA-bB*bK1));
% bL_K2 = (-bRw + bD'*bPK2*bD)\(bD'*bPK2*(bA-bB*bK2));
% bL_K3 = (-bRw + bD'*bPK3*bD)\(bD'*bPK3*(bA-bB*bK3));
cK = min(eig(bRw - bD'*my_bPK*bD))
JK = trace(my_bPK)
% my_cK = min(eig(bRw - bD'*my_bPK*bD))
% [V, D] = eig(bRw - bD'*my_bPK*bD);
% my_JK = trace(my_bPK)
% 
% bPKL1 = Solve_Lya(N, bA, bB, bD, bQ, bRu, bRw, bK, bL1);
% bPKL2 = Solve_Lya(N, bA, bB, bD, bQ, bRu, bRw, bK, bL2);
% bPKL3 = Solve_Lya(N, bA, bB, bD, bQ, bRu, bRw, bK, bL3);
% 
% bPKL4 = Solve_Lya(N, bA, bB, bD, bQ, bRu, bRw, bK1, bL);
% bPKL5 = Solve_Lya(N, bA, bB, bD, bQ, bRu, bRw, bK2, bL);
% bPKL6 = Solve_Lya(N, bA, bB, bD, bQ, bRu, bRw, bK3, bL);
% 
% bPKL7 = Solve_Lya(N, bA, bB, bD, bQ, bRu, bRw, bK1, bL_K1);
% bPKL8 = Solve_Lya(N, bA, bB, bD, bQ, bRu, bRw, bK2, bL_K2);
% bPKL9 = Solve_Lya(N, bA, bB, bD, bQ, bRu, bRw, bK3, bL_K3);

% J1 = trace(bPK1);
% J2 = trace(bPK2);
% J3 = trace(bPK3);
% 
% J4 = trace(bPKL4);
% J5 = trace(bPKL5);
% J6 = trace(bPKL6);
% 
% J7 = trace(bPKL7);
% J8 = trace(bPKL8);
% J9 = trace(bPKL9);

% (J1 + J2)/2-J3
% (J4+J5)/2 - J6
% (J7+J8)/2 - J9
% if cK1 >0 & cK2 >0 & cK3 >0 & ((J1+J2)/2 - J3)<0
%     return;
% end
% if cK > 0
%     return;
% end
% end
