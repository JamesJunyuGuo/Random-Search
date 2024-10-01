%% Formulation
clear all
clc 
close all

rng(77);

% Define Model Parameters (LTI)
N = 9; % length of horizon
m = 3;  % dimension of state
d = 3;  % dimension of control input u
n = 3;  % dimension of the disturbance input w
l = 3;  % dimension of the controlled output z
r = 1e-5; % perturbation radius
s = m*(N+1); % length of concatenated state bX
M = s; % Number of samples

A = [1 0 -10;-1 1 0;0 0 1];
B = [1 -10 0;0 1 0;-1 0 1];
C = eye(m);
D = [0.2 -0.1 0;-0.1 0.2 -0.1;0 -0.1 0.2];
Q = [2 -1 0;-1 2 -1;0 -1 2];
R = [5 -3 0;-3 5 -2;0 -2 5];
W = 1e-2*eye(n);

K = [0.19,0.09,-0.21;-0.20,0.03,-0.12;0.04,0.22,0.07]; % init K to be static

% Initialize Compact Matrices
bX = zeros((N+1)*m, 1);
bU = zeros(N*d, 1);
bA = zeros((N+1)*m, (N+1)*m);
bB = zeros((N+1)*m, N*d);
bC = zeros((N+1)*l, (N+1)*m);
bD = zeros((N+1)*m, N*n);
bQ = zeros((N+1)*m, (N+1)*m);
bR = zeros(N*d, N*d);
bK = zeros(N*d, (N+1)*m);
bKmask = zeros(N*d, (N+1)*m);
bW = zeros(N*n, N*n);
bP = zeros((N+1)*m, (N+1)*m);

% Fill Compact Matrices
for i = 0:N
    bQ(i*m+1:(i+1)*m, i*m+1:(i+1)*m) = Q;
    bW(i*n+1:(i+1)*n, i*n+1:(i+1)*n) = W;
    bC(i*l+1:(i+1)*l, i*m+1:(i+1)*m) = C;
    if i~=N
      bA((i+1)*m+1:(i+2)*m, i*m+1:(i+1)*m) = A;
      bB((i+1)*m+1:(i+2)*m, i*d+1:(i+1)*d) = B;
      bD((i+1)*m+1:(i+2)*m, i*n+1:(i+1)*n) = D;
      bR(i*d+1:(i+1)*d, i*d+1:(i+1)*d) = R;
      bK(i*d+1:(i+1)*d, i*m+1:(i+1)*m) = K; % K init to be static
      bKmask(i*d+1:(i+1)*d, i*m+1:(i+1)*m) = ones(d, m);
    end
end

% Define Algorithm Parameters
max_iter = 400000-1;
etaPG = 1e-7;

% Initialize
sys = ss((bA-bB*bK),(bW)^(1/2),(bQ+bK'*bR*bK)^(1/2)*bC,zeros(size(bW)),[]);
Hinfnorm = norm(sys,Inf,1e-6);
gamma = 0.9*(1/Hinfnorm)^2;
Hinfbnd = 1/sqrt(gamma);

x0 = normrnd(0,1,[1,3])';
w = [x0; normrnd(0,1,[1,n*N])'];
% bX = UpdateFiniteX(x0, m, d, N, bA, bB, bK, w);
[bP, bPt] = Solve_Mixed_P(N, bA, bB, bW, bQ, bR, bK, gamma);
% Sigma0 = eye(n*(N+1));
Jinit = -1*log(det(eye(size(bP))-gamma*bP*bW))/gamma;

% Exact Gradient
% Delta_K = dlyap(((eye(size(bP))-gamma*bP*bW)')\(bA-bB*bK),bW/(eye(size(bP))-gamma*bP*bW));
% full_grad = 2*((bB'*bPt*bB+bR)*bK-bB'*bPt*bA)*Delta_K;

% Finite Diff Check
% e = zeros(size(bK));
% eps = 1e-5;
% e(4, 5) = rand(1);
% eps_K1 = eps*e;eps_K2 = eps*e;
% bKpri1 = bK+eps_K1;bKpri2 = bK-eps_K2;
% [bPup, bPtup] = Solve_Mixed_P(N, bA, bB, bW, bQ, bR, bKpri1, gamma);
% [bPdown, bPtdown] = Solve_Mixed_P(N, bA, bB, bW, bQ, bR, bKpri2, gamma);
% J_eps_bK_1 = -1*log(det(eye(size(bPup))-gamma*bPup*bW))/gamma;
% J_eps_bK_2 = -1*log(det(eye(size(bPdown))-gamma*bPdown*bW))/gamma;
% est_grad = (J_eps_bK_1-J_eps_bK_2)./(eps_K1+eps_K2)

% Optimal Solution
bPs = Solve_Mixed_DARE(N, bA, bB, bW, bQ, bR, gamma);
Jop = -1*log(det(eye(size(bPs))-gamma*bPs*bW))/gamma;

%% Policy Gradient
costPG = zeros(max_iter+1, 1);
HinfPG = zeros(max_iter+1, 1);
costPG(1) = Jinit-Jop;
HinfPG(1) = Hinfnorm;
for i = 1:max_iter
    if mod(i, 1000)==0
        i
    end
    PG = GradEst(bK, bKmask, bA, bB, bQ, bR, bW, gamma, d, n, N, M, r);
    bK = bK - etaPG*PG;
    [bP, bPt] = Solve_Mixed_P(N, bA, bB, bW, bQ, bR, bK, gamma);
    costPG(i+1) = -1*log(det(eye(size(bP))-gamma*bP*bW))/gamma - Jop;
%     sys = ss((bA-bB*bK),(bW)^(1/2),(bQ+bK'*bR*bK)^(1/2)*bC,zeros(size(bW)),[]);
%     HinfPG(i+1) = norm(sys,Inf,1e-6);
end

%% Plot Results
linewidth = 3;

figure
semilogy(0:max_iter,(costPG(:)),'linewidth',linewidth);
title('Cost Error in Exact PG', 'FontSize', 80)
legend('Policy Gradient','Interpreter','latex','Location','northeast', 'FontSize',25)
xlabel('Iterations','FontSize',80)
ylabel('$\mathcal{J}(\textbf{K})-\mathcal{J}(\textbf{K}^*)$','Interpreter','latex', 'FontSize',80)
grid on
set(gca,'FontSize', 20);

% figure
% semilogy(0:max_iter,Hinfbnd*ones(max_iter+1,1),'k-*','linewidth',linewidth);
% hold on;
% semilogy(0:max_iter,(HinfPG(:)),'r-*','linewidth',linewidth);
% title('Hinf norm in Exact PG', 'FontSize', 80)
% legend('Policy Gradient','Interpreter','latex','Location','northeast', 'FontSize',25)
% xlabel('Iterations','FontSize',80)
% ylabel('$H_{\infty}$-norm','Interpreter','latex', 'FontSize',80)
% grid on
% set(gca,'FontSize', 20);