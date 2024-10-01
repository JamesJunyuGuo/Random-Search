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
D = [0.5 0 0; 0 0.2 0; 0 0 0.2];
Q = [2 -1 0;
     -1 2 -1;
     0 -1 2]; % Q, R has to be symmetric and Q>=0, R>0
Ru = [4 -1 0;
      -1 4 -2;
      0 -2 3];
Rw = 5*eye(3);

% initialize a K
K = [-0.08 0.35 0.62; -0.21 0.19 0.32; -0.06 0.10 0.41];
% 
% K = [-0.23    0.14    0.34;
%    -0.16    0.07    0.16;
%    -0.06    0.05    0.63];
% cKsave = 3.2325;
% for h = 1:10000000
% e = normrnd(0, 0.01, [3, 3]);
% K = K+e;
% K = [-0.04 -0.01 0.61; -0.21 0.15 0.15; -0.06 0.05 0.42];
L = [0.03 0.35 0.62; -0.21 0.19 0.32; -0.06 0.10 0.41]

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
bL = zeros(N*n, (N+1)*m);
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
      bL(i*n+1:(i+1)*n, i*m+1:(i+1)*m) = L;
      bKmask(i*d+1:(i+1)*d, i*m+1:(i+1)*m) = ones(d, m);
      bLmask(i*n+1:(i+1)*n, i*m+1:(i+1)*m) = ones(n, m);
    end
end

% Define Algorithm Parameters
K_iter1 = 100;
K_iter2 = 1500;
K_iter2hard = 10000;
K_iter3 = 100;
K_iter3_hard = 500;
% etaNPG_L = 0.1;
% alphaNPG_K = 1e-5;
% M1 = 100;
M1 = 100000;
M2 = 3000000;
r1 = 1;
r2 = 0.03;
var = 0.05;
eps1 = 0.0001;
eps2 = 0.8;
% eps1 = 0.02;
% Exact Gradient
bPKL = Solve_Lya(N, bA, bB, bD, bQ, bRu, bRw, bK, bL);
NPG_K = 2*((bRu + bB'*bPKL*bB)*bK-bB'*bPKL*(bA-bD*bL));
NPG_L = 2*((-bRw + bD'*bPKL*bD)*bL - bD'*bPKL*(bA-bB*bK));
Sigma0 = var*eye(n*(N+1));
Sigma_KL = SolveSigmaL(N, bA, bB, bD, bK, bL, Sigma0);
Exact_Grad_L = NPG_L*Sigma_KL;
Exact_Grad_K = NPG_K*Sigma_KL;

% solve GARE
[X, L, G] = dare(bA, [bB bD], bQ, [bRu, zeros(size(bRu)); zeros(size(bRu)) -bRw]);
Kop = (inv(bRu+bB'*(X-X*bD*inv(-bRw + bD'*X*bD)*bD'*X)*bB)*bB'*X*(bA-bD*inv(-bRw + bD'*X*bD)*bD'*X*bA)).*bKmask;
Kop2 = inv(bRu)*(bB'*X*inv(eye(9)+(bB*inv(bRu)*bB' - bD*inv(bRw)*bD')*X)*bA);
Lop = (inv(-bRw + bD'*(X-X*bB*inv(bRu + bB'*X*bB)*bB'*X)*bD)*bD'*X*(bA-bB*inv(bRu+bB'*X*bB)*bB'*X*bA)).*bLmask;
Lop2 = -inv(bRw)*(bD'*X*inv(eye(9)+(bB*inv(bRu)*bB' - bD*inv(bRw)*bD')*X)*bA);
Jop = trace(Solve_Lya(N, bA, bB, bD, bQ, bRu, bRw, Kop, Lop)*Sigma0);

% test for K estimator
% bPK = Solve_Riccati_K(N, bA, bB, bD, bQ, bRu, bRw, bK);
% bLs = (-bRw + bD'*bPK*bD)\(bD'*bPK*(bA-bB*bK));
% bPKLs = Solve_Lya(N, bA, bB, bD, bQ, bRu, bRw, bK, bLs);
% NPG_K = 2*((bRu + bB'*bPK*bB)*bK-bB'*bPK*(bA-bD*bLs));
% Sigma_K = SolveSigmaL(N, bA, bB, bD, bK, bLs, Sigma0);
% PG_K = NPG_K*Sigma_K;
% 
% etaNPG_L = 8e-2;
% 
% [Est_PG_K, Est_Sigma_K] = OuterGradEst(bK, bLs, bKmask, bLmask, bA, bB, bD, bQ, bRu, bRw, Sigma0, M1, M2, m, n, d, N, r1, r2, var, eps1, etaNPG_L);
% Est_NPG_K = Est_PG_K/Est_Sigma_K;
% 
% norm(Est_PG_K - PG_K)
% norm(Est_Sigma_K - Sigma_K)
% norm(Est_NPG_K - NPG_K)

% trace(Solve_Lya(N, bA, bB, bD, bQ, bRu, bRw, bK, bLs)*Sigma0)
% estimate grad L
% [Est_GradL, Est_CovL] = InnerGradEst(bK, bL, bLmask, bA, bB, bD, bQ, bRu, bRw, Sigma0, M1, m, n, N, r1, var);
% Est_GradL;
% Exact_Grad_L;
% norm(Exact_Grad_L - Est_GradL)
% norm(Sigma_KL - Est_CovL)
% norm(NPG_L - Est_GradL*inv(Est_CovL))

% estimate natural grad K
% [Est_GradK, Est_CovK] = OuterGradEst(bK, bL, bKmask, bA, bB, bD, bQ, bRu, bRw, Sigma0, M2, m, d, N, r2, var);
% norm(Exact_Grad_K - Est_GradK)
% norm(Sigma_KL - Est_CovK)
% norm(NPG_K - Est_GradK*inv(Est_CovK))

% SigmaL = dlyap((bA-bB*bK-bD*bL), Sigma0) 10 times slower

% Finite Diff Check
% e = zeros(size(bL));
% eps = 1e-5;
% e(2, 2) = rand(1);
% eps_L = eps*e;
% bLup = bL+eps_L;bLdown = bL-eps_L;
% bPup = Solve_Lya(N, bA, bB, bD, bQ, bRu, bRw, bK, bLup);
% bPdown = Solve_Lya(N, bA, bB, bD, bQ, bRu, bRw, bK, bLdown);
% est_PG = (trace(bPup*Sigma0) - trace(bPdown*Sigma0))./(2*eps_L)


bPK = Solve_Riccati_K(N, bA, bB, bD, bQ, bRu, bRw, bK); % for a fixed K, solve the Riccati of opt wrt L
etaPG_L = 0.1;
etaNPG_L = 1/(2*norm(bRw-bD'*bPKL*bD));
etaGN_L = 0.5;
alphaNPG_K1 = 1e-4;
alphaNPG_K2 = 1/(2*norm(bRu+bB'*bPK*bB))/10;
alphaNPG_K2hard = 1/(2*norm(bRu+bB'*bPK*bB))/1000;
alphaGN_K3 = 0.5/1000;
alphaGN_K3_hard = 0.5/2000;
% bPL = Solve_Riccati_L(N, bA, bB, bD, bQ, bRu, bRw, bL); % for a fixed L, solve the Riccati of opt wrt K
cK = min(eig(bRw - bD'*bPK*bD));
% if(cK > cKsave)
%     cKsave = cK
% else
%     K = K-e;
% end
% end
% cL = min(eig(bRu + bB'*bPL*bB))
Jinit = trace(bPKL*Sigma0);

%% PG-NPG
cost1 = zeros(100000000, 1);
cost1(1) = Jinit;
gradnormL1 = zeros(100000000, 1);
gradnormK1 = zeros(100000000, 1);
cK1 = zeros(100000000, 1);
cK1(1) = min(eig(bRw - bD'*bPKL*bD));
old_idx1 = 1;
bK1 = bK;
bL1 = bL;
gradnormK1sum = 0;
for i = 1:K_iter1
    i
    idx = old_idx1;
    % calculate how well the inner loop can do
    bestL = trace(Solve_Riccati_K(N, bA, bB, bD, bQ, bRu, bRw, bK1)*Sigma0);
    currL = trace(Solve_Lya(N, bA, bB, bD, bQ, bRu, bRw, bK1, bL1)*Sigma0);
    gradNormL1sum = 0;
    while bestL > currL + eps1
       bestL - currL
       bPKL = Solve_Lya(N, bA, bB, bD, bQ, bRu, bRw, bK1, bL1);
       Sigma_L = SolveSigmaL(N, bA, bB, bD, bK1, bL1, Sigma0);
       PG_L = 2*((-bRw + bD'*bPKL*bD)*bL1 - bD'*bPKL*(bA-bB*bK1))*Sigma_L;
       [Est_PG_L, Est_Sigma_L] = InnerGradEst(bK1, bL1, bLmask, bA, bB, bD, bQ, bRu, bRw, Sigma0, M1, m, n, N, r1, var);
       %calculate average grad norm
       norm(PG_L-Est_PG_L);
%        Est_NPG_L = Est_PG_L * inv(Est_Sigma_L);
       gradNormL1sum = gradNormL1sum + norm(Est_PG_L, 'fro');
       gradnormL1(idx) = gradNormL1sum/(idx+1-old_idx1);
       gradnormK1(idx) = inf;
       bL1 = bL1 + etaPG_L*Est_PG_L;
       idx = idx + 1;
       bPKL = Solve_Lya(N, bA, bB, bD, bQ, bRu, bRw, bK1, bL1);
       cK1(idx) = min(eig(bRw - bD'*bPKL*bD));
       currL = trace(bPKL*Sigma0);
       cost1(idx) = currL;
    end
    gradnormL1(idx) = inf;
    gradnormK1sum = gradnormK1sum +(norm(NPG_K*Sigma_L, 'fro'));
    gradnormK1(idx) = gradnormK1sum/i;
    idx = idx+1;
    old_idx1 = idx;
    bPKL = Solve_Lya(N, bA, bB, bD, bQ, bRu, bRw, bK1, bL1);
    NPG_K = 2*((bRu + bB'*bPKL*bB)*bK1-bB'*bPKL*(bA-bD*bL1));
    bK1 = bK1 - alphaNPG_K1*NPG_K;
    %check IR
    bPK = Solve_Riccati_K(N, bA, bB, bD, bQ, bRu, bRw, bK1);
    if min(eig(bRw - bD'*bPK*bD)) < 0
        disp('out of cK');
        return;
    end
    bPKL = Solve_Lya(N, bA, bB, bD, bQ, bRu, bRw, bK1, bL1);
    cK1(idx) = min(eig(bRw - bD'*bPKL*bD));
    cost1(idx) = trace(bPKL*Sigma0);
end

ImageDPI=400;
linewidth = 3;

cost1size = find(cost1==0, 1, 'first')
cK1size = find(cK1==0, 1, 'first')
figure
hold on;
yyaxis left
semilogy(0:cost1size-2,(cost1(1:cost1size-1)),'-','color',[0, 0.4470, 0.7410], 'linewidth',linewidth,'markersize', 1);
ylabel('$\mathcal{G}(\bf{K}, \bf{L})$','Interpreter','latex', 'FontSize',20)
yyaxis right
plot(0:cK1size-2,(cK1(1:cK1size-1)),'-','color',[0.8500, 0.3250, 0.0980], 'linewidth',linewidth, 'markersize', 1);
ylabel('$\lambda_{\min}(\bf{R}^w-\bf{D}^{\top}\bf{P}_{\bf{K}, \bf{L}}\bf{D})$','Interpreter','latex', 'FontSize',20)
legend('PG-NPG, $\mathcal{G}$','PG-NPG, $\lambda_{\min}$','Interpreter','latex','Location','northeast', 'FontSize',18)
xlabel('Total Iterations $K \times L$','Interpreter','latex' ,'FontSize',20)
set(gca,'FontSize', 18);



figure
hold on;
gradnormL1size = find(gradnormL1==0, 1, 'first');
gradnormK1size = find(gradnormK1==0, 1, 'first');
yyaxis left
ylabel('Avg. $|\nabla_{\bf{L}}|_F$','Interpreter','latex', 'FontSize',20)
semilogy(0:gradnormL1size-2,(gradnormL1(1:gradnormL1size-1)),'-','color',[0.4940, 0.1840, 0.5560], 'linewidth',linewidth, 'markersize', 13);
yyaxis right
semilogy(0:gradnormK1size-2,(gradnormK1(1:gradnormK1size-1)),'.','color',[0.4660, 0.6740, 0.1880], 'linewidth',linewidth, 'markersize', 13);
legend('PG-NPG, Avg.$|\nabla_{\bf{L}}|_F$','PG-NPG, Avg.$|\nabla_{\bf{K}}|_F$', 'Interpreter','latex','Location','northeast', 'FontSize',18)
xlabel('Total Iterations $K \times L$','Interpreter','latex' ,'FontSize',20)
ylabel('Avg. $|\nabla_{\bf{K}}|_F$','Interpreter','latex', 'FontSize',20)
set(gca,'FontSize', 18);

%% NPG-NPG
cost1 = zeros(10000, 1);
cost1(1) = Jinit;
gradnormL1 = zeros(10000, 1);
gradnormK1 = zeros(10000, 1);
cK1 = zeros(10000, 1);
cK1(1) = min(eig(bRw - bD'*bPKL*bD));
old_idx1 = 1;
bK1 = bK;
bL1 = bL;
gradnormK1sum = 0;
currK = Jinit;
idx = 1;
while currK > Jop + eps2
    % solve approximate inner loop best response
%     bL1 = Inner_Oracle_ZONPG(bK1, bL1, bLmask, bA, bB, bD, bQ, bRu, bRw, Sigma0, M1, m, n, N, r1, var, eps1, etaNPG_L);
    bL1 = Inner_Oracle_FONPG(bK1, bL1, N, bA, bB, bD, bQ, bRu, bRw, Sigma0, eps1, etaNPG_L);
    
    bPKL = Solve_Lya(N, bA, bB, bD, bQ, bRu, bRw, bK1, bL1);
%     NPG_K = 2*((bRu + bB'*bPKL*bB)*bK1-bB'*bPKL*(bA-bD*bL1));
    
    [Est_PG_K, Est_Sigma_K] = OuterGradEst(bK, bL, bKmask, bLmask, bA, bB, bD, bQ, bRu, bRw, Sigma0, M1, M2, m, n, d, N, r1, r2, var, eps1, etaNPG_L);
    Est_NPG_K = Est_PG_K/Est_Sigma_K;
    
%     norm(NPG_K - Est_NPG_K)
    
%     bK1 = bK1 - alphaNPG_K1*Est_NPG_K;
    bK1 = bK1 - alphaNPG_K1*NPG_K;
    %check IR
    bPK = Solve_Riccati_K(N, bA, bB, bD, bQ, bRu, bRw, bK1);
    if min(eig(bRw - bD'*bPK*bD)) < 0
        disp('out of cK');
        return;
    end
%     bPKL = Solve_Lya(N, bA, bB, bD, bQ, bRu, bRw, bK1, bL1);
    cK1(idx+1) = min(eig(bRw - bD'*bPK*bD));
    cost1(idx+1) = trace(bPK*Sigma0);
    currK = cost1(idx+1)
    idx = idx+1;
end

ImageDPI=400;
linewidth = 3;

cost1size = find(cost1==0, 1, 'first')
cK1size = find(cK1==0, 1, 'first')
figure
hold on;
yyaxis left
semilogy(0:cost1size-2,(cost1(1:cost1size-1)),'-','color',[0, 0.4470, 0.7410], 'linewidth',linewidth,'markersize', 1);
ylabel('$\mathcal{G}(\bf{K}, \bf{L})$','Interpreter','latex', 'FontSize',20)
yyaxis right
plot(0:cK1size-2,(cK1(1:cK1size-1)),'-','color',[0.8500, 0.3250, 0.0980], 'linewidth',linewidth, 'markersize', 1);
ylabel('$\lambda_{\min}(\bf{R}^w-\bf{D}^{\top}\bf{P}_{\bf{K}, \bf{L}}\bf{D})$','Interpreter','latex', 'FontSize',20)
legend('PG-NPG, $\mathcal{G}$','PG-NPG, $\lambda_{\min}$','Interpreter','latex','Location','northeast', 'FontSize',18)
xlabel('Total Iterations $K \times L$','Interpreter','latex' ,'FontSize',20)
set(gca,'FontSize', 18);



figure
hold on;
gradnormL1size = find(gradnormL1==0, 1, 'first');
gradnormK1size = find(gradnormK1==0, 1, 'first');
yyaxis left
ylabel('Avg. $|\nabla_{\bf{L}}|_F$','Interpreter','latex', 'FontSize',20)
semilogy(0:gradnormL1size-2,(gradnormL1(1:gradnormL1size-1)),'-','color',[0.4940, 0.1840, 0.5560], 'linewidth',linewidth, 'markersize', 13);
yyaxis right
semilogy(0:gradnormK1size-2,(gradnormK1(1:gradnormK1size-1)),'.','color',[0.4660, 0.6740, 0.1880], 'linewidth',linewidth, 'markersize', 13);
legend('PG-NPG, Avg.$|\nabla_{\bf{L}}|_F$','PG-NPG, Avg.$|\nabla_{\bf{K}}|_F$', 'Interpreter','latex','Location','northeast', 'FontSize',18)
xlabel('Total Iterations $K \times L$','Interpreter','latex' ,'FontSize',20)
ylabel('Avg. $|\nabla_{\bf{K}}|_F$','Interpreter','latex', 'FontSize',20)
set(gca,'FontSize', 18);


%% ZONPG-FONPG
cost2 = zeros(100000, 1);
cost2(1) = Jinit;
gradnormL2 = zeros(100000, 1);
gradnormK2 = zeros(100000, 1);
cK2 = zeros(100000, 1);
cK2(1) = min(eig(bRw - bD'*bPKL*bD));
old_idx2 = 1;
bK2 = bK;
bL2 = bL;
gradnormK2sum = 0;
currK = Jinit;
while currK > Jop + eps2
    idx = old_idx2;
    % calculate how well the inner loop can do
    bestL = trace(Solve_Riccati_K(N, bA, bB, bD, bQ, bRu, bRw, bK2)*Sigma0);
    currL = trace(Solve_Lya(N, bA, bB, bD, bQ, bRu, bRw, bK2, bL2)*Sigma0);
    gradNormL2sum = 0;
    while bestL > currL + eps1
       bPKL = Solve_Lya(N, bA, bB, bD, bQ, bRu, bRw, bK2, bL2);
       SigmaL = SolveSigmaL(N, bA, bB, bD, bK2, bL2, Sigma0);
%        NPG_L = 2*((-bRw + bD'*bPKL*bD)*bL2 - bD'*bPKL*(bA-bB*bK2));
       [Est_PG_L, Est_Sigma_L] = InnerGradEst(bK2, bL2, bLmask, bA, bB, bD, bQ, bRu, bRw, Sigma0, M1, m, n, N, r1, var);
       Est_NPG_L = Est_PG_L/Est_Sigma_L;
       %calculate average grad norm
       gradNormL2sum = gradNormL2sum + norm(Est_NPG_L, 'fro');
       gradnormL2(idx) = gradNormL2sum/(idx+1-old_idx2);
       gradnormK2(idx) = inf;
       bL2 = bL2 + etaPG_L*Est_PG_L;
       idx = idx + 1;
       bPKL = Solve_Lya(N, bA, bB, bD, bQ, bRu, bRw, bK2, bL2);
       cK2(idx) = min(eig(bRw - bD'*bPKL*bD));
       currL = trace(bPKL*Sigma0)
       cost2(idx) = currL;
    end
    gradnormL2(idx) = inf;
    gradnormK2sum = gradnormK2sum +(norm(NPG_K, 'fro'));
    gradnormK2(idx) = gradnormK2sum/i;
    idx = idx+1;
    old_idx2 = idx;
    bPKL = Solve_Lya(N, bA, bB, bD, bQ, bRu, bRw, bK2, bL2);
    NPG_K = 2*((bRu + bB'*bPKL*bB)*bK2-bB'*bPKL*(bA-bD*bL2));
    bK2 = bK2 - alphaNPG_K2*NPG_K;
    %check IR
    bPK = Solve_Riccati_K(N, bA, bB, bD, bQ, bRu, bRw, bK2);
    if min(eig(bRw - bD'*bPK*bD)) < 0
        disp('out of cK');
        return;
    end
    bPKL = Solve_Lya(N, bA, bB, bD, bQ, bRu, bRw, bK2, bL2);
    cK2(idx) = min(eig(bRw - bD'*bPKL*bD));
    cost2(idx) = trace(bPKL*Sigma0);
    currK = cost2(idx)
end

ImageDPI=400;
linewidth = 3;
cost2size = find(cost2==0, 1, 'first')
cK2size = find(cK2==0, 1, 'first')
figure
hold on;
yyaxis left
semilogy(0:cost2size-2,(cost2(1:cost2size-1)),'-','color',[0, 0.4470, 0.7410], 'linewidth',linewidth,'markersize', 1);
% yticks([200, 400, 600])
ylabel('$\mathcal{G}(\bf{K}, \bf{L})$','Interpreter','latex', 'FontSize',20)
yyaxis right
plot(0:cK2size-2,(cK2(1:cK2size-1)),'-','color',[0.8500, 0.3250, 0.0980], 'linewidth',linewidth, 'markersize', 1);
ylabel('$\lambda_{\min}(\bf{R}^w-\bf{D}^T\bf{P}_{\bf{K}, \bf{L}}\bf{D})$','Interpreter','latex', 'FontSize',20)
legend('NPG-NPG, $\mathcal{G}$','NPG-NPG, $\lambda_{\min}$','Interpreter','latex','Location','northeast', 'FontSize',18)
xlabel('Total Iterations $K \times L$','Interpreter','latex' ,'FontSize',20)
set(gca,'FontSize', 18);

figure
hold on;
gradnormL2size = find(gradnormL2==0, 1, 'first');
gradnormK2size = find(gradnormK2==0, 1, 'first');
yyaxis left
ylabel('Avg. $|\nabla_{\bf{L}}|_F$','Interpreter','latex', 'FontSize',20)
semilogy(0:gradnormL2size-2,(gradnormL2(1:gradNormL2size-1)),'-','color',[0.4940, 0.1840, 0.5560], 'linewidth',linewidth, 'markersize', 13);
% yticks([20, 40, 60, 80, 100, 120])
yyaxis right
semilogy(0:gradnormK2size-2,(gradnormK2(1:gradnormK2size-1)),'.','color',[0.4660, 0.6740, 0.1880], 'linewidth',linewidth, 'markersize', 13);
legend('NPG-NPG, Avg.$|\nabla_{\bf{L}}|_F$','NPG-NPG, Avg.$|\nabla_{\bf{K}}|_F$', 'Interpreter','latex','Location','northeast', 'FontSize',18)
xlabel('Total Iterations $K \times L$','Interpreter','latex' ,'FontSize',20)
ylabel('Avg. $|\nabla_{\bf{K}}|_F$','Interpreter','latex', 'FontSize',20)
set(gca,'FontSize', 18);


%% FONPG-ZONPG
cost2 = zeros(100000, 1);
cost2(1) = Jinit;
gradnormL2 = zeros(100000, 1);
gradnormK2 = zeros(100000, 1);
cK2 = zeros(100000, 1);
cK2(1) = min(eig(bRw - bD'*bPKL*bD));
old_idx2 = 1;
bK2 = bK;
bL2 = bL;
gradnormK2sum = 0;
currK = Jinit;
while currK > Jop + eps2
    idx = old_idx2;
    % calculate how well the inner loop can do
    bestL = trace(Solve_Riccati_K(N, bA, bB, bD, bQ, bRu, bRw, bK2)*Sigma0)
    currL = trace(Solve_Lya(N, bA, bB, bD, bQ, bRu, bRw, bK2, bL2)*Sigma0);
    gradNormL2sum = 0;
    while bestL > currL + eps1
       bPKL = Solve_Lya(N, bA, bB, bD, bQ, bRu, bRw, bK2, bL2);
       SigmaL = SolveSigmaL(N, bA, bB, bD, bK2, bL2, Sigma0);
       NPG_L = 2*((-bRw + bD'*bPKL*bD)*bL2 - bD'*bPKL*(bA-bB*bK2));
%        [Est_PG_L, Est_Sigma_L] = InnerGradEst(bK2, bL2, bLmask, bA, bB, bD, bQ, bRu, bRw, Sigma0, M1, m, n, N, r1, var);
%        Est_NPG_L = Est_PG_L/Est_Sigma_L;
       %calculate average grad norm
%        gradNormL2sum = gradNormL2sum + norm(Est_NPG_L, 'fro');
%        gradnormL2(idx) = gradNormL2sum/(idx+1-old_idx2);
%        gradnormK2(idx) = inf;
       % PG
       bL2 = bL2 + etaNPG_L*NPG_L;
       idx = idx + 1;
       bPKL = Solve_Lya(N, bA, bB, bD, bQ, bRu, bRw, bK2, bL2);
       cK2(idx) = min(eig(bRw - bD'*bPKL*bD));
       currL = trace(bPKL*Sigma0)
       cost2(idx) = currL;
    end
%     disp('hi')
    gradnormL2(idx) = inf;
    gradnormK2sum = gradnormK2sum +(norm(NPG_K, 'fro'));
    gradnormK2(idx) = gradnormK2sum/i;
    idx = idx+1;
    old_idx2 = idx;
    bPKL = Solve_Lya(N, bA, bB, bD, bQ, bRu, bRw, bK2, bL2);
    [Est_PG_K, Est_Sigma_K] = OuterGradEst(bK2, bL2, bKmask, bLmask, bA, bB, bD, bQ, bRu, bRw, Sigma0, M1, M2, m, n, d, N, r1, r2, var, eps1, etaNPG_L);
    Est_NPG_K = Est_PG_K/Est_Sigma_K;
    bK2 = bK2 - alphaNPG_K2*Est_NPG_K;
    %check IR
    bPK = Solve_Riccati_K(N, bA, bB, bD, bQ, bRu, bRw, bK2);
    if min(eig(bRw - bD'*bPK*bD)) < 0
        disp('out of cK');
        return;
    end
    bPKL = Solve_Lya(N, bA, bB, bD, bQ, bRu, bRw, bK2, bL2);
    cK2(idx) = min(eig(bRw - bD'*bPKL*bD));
    cost2(idx) = trace(bPKL*Sigma0);
    currK = cost2(idx)
end

ImageDPI=400;
linewidth = 3;
cost2size = find(cost2==0, 1, 'first')
cK2size = find(cK2==0, 1, 'first')
figure
hold on;
yyaxis left
semilogy(0:cost2size-2,(cost2(1:cost2size-1)),'-','color',[0, 0.4470, 0.7410], 'linewidth',linewidth,'markersize', 1);
% yticks([200, 400, 600])
ylabel('$\mathcal{G}(\bf{K}, \bf{L})$','Interpreter','latex', 'FontSize',20)
yyaxis right
plot(0:cK2size-2,(cK2(1:cK2size-1)),'-','color',[0.8500, 0.3250, 0.0980], 'linewidth',linewidth, 'markersize', 1);
ylabel('$\lambda_{\min}(\bf{R}^w-\bf{D}^T\bf{P}_{\bf{K}, \bf{L}}\bf{D})$','Interpreter','latex', 'FontSize',20)
legend('NPG-NPG, $\mathcal{G}$','NPG-NPG, $\lambda_{\min}$','Interpreter','latex','Location','northeast', 'FontSize',18)
xlabel('Total Iterations $K \times L$','Interpreter','latex' ,'FontSize',20)
set(gca,'FontSize', 18);

