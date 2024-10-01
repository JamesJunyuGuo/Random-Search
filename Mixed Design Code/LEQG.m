%% Formulation
clear all
clc 
close all

% rng(77);

% Define Model Parameters (LTI)
N = 2; % length of horizon
m = 3;  % dimension of state
d = 3;  % dimension of control input u
n = 3;  % dimension of the disturbance input w
l = 3;  % dimension of the controlled output z

A = [1 0 -5;-1 1 0;0 0 1];
B = [1 -10 0;0 3 1;-1 0 2];
Q = [2 -1 0;
     -1 2 -1;
     0 -1 2]; % Q, R has to be symmetric and Q>=0, R>0
R = [4 -1 0;
      -1 4 -2;
      0 -2 3];
W = 0.1*eye(3);
beta = 0.2;
I = eye(3);
Z = zeros(3);
% initialize a K
K = [-0.08 0.35 0.62; -0.21 0.19 0.32; -0.06 0.10 0.41];

% Initialize Compact Matrices
bX = zeros((N+1)*m, 1);
bU = zeros(N*d, 1);
bA = zeros((N+1)*m, (N+1)*m);
bB = zeros((N+1)*m, N*d);
bQ = zeros((N+1)*m, (N+1)*m);
bR = zeros(N*d, N*d);
bK = zeros(N*d, (N+1)*m);
bP = zeros((N+1)*m, (N+1)*m);
bKmask = zeros(N*d, (N+1)*m);


% Fill Compact Matrices
for i = 0:N
    bQ(i*m+1:(i+1)*m, i*m+1:(i+1)*m) = Q;
    if i~=N
      bA((i+1)*m+1:(i+2)*m, i*m+1:(i+1)*m) = A;
      bB((i+1)*m+1:(i+2)*m, i*d+1:(i+1)*d) = B;
      bR(i*d+1:(i+1)*d, i*d+1:(i+1)*d) = R;
      bK(i*d+1:(i+1)*d, i*m+1:(i+1)*m) = K; 
      bKmask(i*d+1:(i+1)*d, i*m+1:(i+1)*m) = ones(d, m);
    end
end

P2 = Q;
P1 = Q + K'*R*K + (A-B*K)'*(P2 + beta*P2*inv(inv(W)-beta*P2)*P2)*(A-B*K)
P0 = Q + K'*R*K + (A-B*K)'*(P1 + beta*P1*inv(inv(W)-beta*P1)*P1)*(A-B*K)
min(eig(I - beta*P2*W))
min(eig(I - beta*P1*W))
min(eig(I - beta*P0*W))
J0 = -(1/beta)*(log(det(I-beta*P0*W)) + log(det(I-beta*P1*W)) + log(det(I-beta*P2*W)))
bP = [P0 Z Z; Z P1 Z; Z Z P2];
bW = [W Z Z; Z W Z; Z Z W];
J0_compact = -(1/beta)*log(det(eye(9)-beta*bP*bW));
tildebP = bP+beta*bP*inv(inv(bW) - beta*bP)*bP;
sigma = bW*inv(eye(9) - beta*bP*bW);
sigma2 = sqrt(bW)*inv(eye(9)-beta*sqrt(bW)*bP*sqrt(bW))*sqrt(bW);
Grad_compact = 2*((bR+bB'*tildebP*bB)*bK - bB'*tildebP*bA) * sigma

tildeP1 = P1+beta*P1*inv(inv(W) - beta*P1)*P1;
sigma0 = W*inv(I - beta*P0*W);
GradK0 = 2*((R+B'*tildeP1*B)*K - B'*tildeP1*A)*sigma0;

tildeP2 = P2+beta*P2*inv(inv(W) - beta*P2)*P2;
sigma1 = W*inv(I - beta*P1*W);
sigma1 = sigma1 + inv(I-beta*P1*W)'*(A-B*K)*sqrt(W)*inv(I - beta*sqrt(W)*P0*sqrt(W))*sqrt(W)*(A-B*K)'*inv(I-beta*P1*W);
GradK1 = 2*((R+B'*tildeP2*B)*K - B'*tildeP2*A)*sigma1


% Finite Diff Check
e = zeros(size(K));
eps = 1e-5;
e(1, 1) = rand(1);
eps_K = eps*e;
K0up = K; K0down = K;
K1up = K+eps_K; K1down = K-eps_K;

P1up = Q + K1up'*R*K1up + (A-B*K1up)'*(P2 + beta*P2*inv(inv(W)-beta*P2)*P2)*(A-B*K1up);
P0up = Q + K0up'*R*K0up + (A-B*K0up)'*(P1up + beta*P1up*inv(inv(W)-beta*P1up)*P1up)*(A-B*K0up);

P1down = Q + K1down'*R*K1down + (A-B*K1down)'*(P2 + beta*P2*inv(inv(W)-beta*P2)*P2)*(A-B*K1down);
P0down = Q + K0down'*R*K0down + (A-B*K0down)'*(P1down + beta*P1down*inv(inv(W)-beta*P1down)*P1down)*(A-B*K0down);

cost_up = -(1/beta)*(log(det(I-beta*P0up*W)) + log(det(I-beta*P1up*W)) + log(det(I-beta*P2*W)));
cost_down = -(1/beta)*(log(det(I-beta*P0down*W)) + log(det(I-beta*P1down*W)) + log(det(I-beta*P2*W)));
est_PG = (cost_up - cost_down)./(2*eps_K)


% Finite Diff Check
% e = zeros(size(bK));
% eps = 1e-5;
% e(5, 5) = rand(1);
% eps_K = eps*e;
% bKup = bK+eps_K; bKdown = bK-eps_K;
% 
% bPK_up = bQ + bKup'*bR*bKup;
% bPK_down = bQ + bKdown'*bR*bKdown;
% for i = 1:N
%     tildebPK_up = bPK_up+beta*bPK_up*inv(inv(bW) - beta*bPK_up)*bPK_up;
%     bPK_up = bQ + bKup'*bR*bKup + (bA-bB*bKup)'*tildebPK_up*(bA-bB*bKup);
%     
%     tildebPK_down = bPK_down+beta*bPK_down*inv(inv(bW) - beta*bPK_down)*bPK_down;
%     bPK_down = bQ + bKdown'*bR*bKdown + (bA-bB*bKdown)'*tildebPK_down*(bA-bB*bKdown);
% end
% cost_up = -(1/beta)*log(det(eye(9)-beta*bPK_up*bW));
% cost_down = -(1/beta)*log(det(eye(9)-beta*bPK_down*bW));
% est_PG = (cost_up - cost_down)./(2*eps_K)
