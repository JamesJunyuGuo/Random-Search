%% Formulation
clear all
clc 
close all

rng(77);

% Define Model Parameters (LTI)
N = 5; % length of horizon
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

% for i = 1:100000000
    
K = 2*(rand(3)-0.5);
L = 2*(rand(3)- 0.5);
K = [-0.14 -0.04 0.62; -0.21 0.14 0.15; -0.06 0.05 0.42];
% K = [-0.12 -0.01 0.62; -0.21 0.14 0.15; -0.06 0.05 0.42];
% K = [-0.1362    0.0934    0.6458;
%    -0.2717   -0.1134   -0.4534;
%    -0.6961   -0.9279   -0.6620];
L = [0.2887   -0.2286    0.4588;
   -0.7849   -0.1089   -0.3755;
   -0.2935    0.9541    0.7895]
% L = [0.8382   -0.7214   -0.3477;
%     0.2844   -0.8254    0.0821;
%     0.5074    0.5760   -0.5195];

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
K_iter1 = 200000;
K_iter2 = 1500;
K_iter2hard = 10000;
K_iter3 = 10000;
K_iter3_hard = 500;
% etaNPG_L = 0.1;
% alphaNPG_K = 1e-5;

% Exact Gradient
bPKL = Solve_Lya(N, bA, bB, bD, bQ, bRu, bRw, bK, bL);
NPG_K = 2*((bRu + bB'*bPKL*bB)*bK-bB'*bPKL*(bA-bD*bL));
NPG_L = 2*((-bRw + bD'*bPKL*bD)*bL - bD'*bPKL*(bA-bB*bK));
Sigma0 = eye(n*(N+1));

SigmaL = SolveSigmaL(N, bA, bB, bD, bK, bL, Sigma0)
% SigmaL = dlyap((bA-bB*bK-bD*bL), Sigma0) 10 times slower

PG_L = NPG_L*SigmaL;

% Finite Diff Check
% e = zeros(size(bL));
% eps = 1e-5;
% e(7, 7) = rand(1);
% eps_L = eps*e;
% bLup = bL+eps_L;bLdown = bL-eps_L;
% bPup = Solve_Lya(N, bA, bB, bD, bQ, bRu, bRw, bK, bLup);
% bPdown = Solve_Lya(N, bA, bB, bD, bQ, bRu, bRw, bK, bLdown);
% est_PG = (trace(bPup) - trace(bPdown))./(2*eps_L)


bPK = Solve_Riccati_K(N, bA, bB, bD, bQ, bRu, bRw, bK); % for a fixed K, solve the Riccati of opt wrt L
etaPG_L = 1e-4;
etaNPG_L = 1/(2*norm(bRw-bD'*bPKL*bD));
etaGN_L = 0.5;
alphaNPG_K1 = 1/(2*norm(bRu+bB'*bPK*bB))/100;
alphaNPG_K2 = 1/(2*norm(bRu+bB'*bPK*bB))/100;
alphaNPG_K2hard = 1/(2*norm(bRu+bB'*bPK*bB))/1000
alphaNPG_Alter = 1/(2*norm(bRu+bB'*bPK*bB))/100;
etaNPG_Alter = alphaNPG_Alter;
alphaGN_K3 = 0.5/1000
alphaGN_K3_hard = 0.5/2000
bPL = Solve_Riccati_L(N, bA, bB, bD, bQ, bRu, bRw, bL); % for a fixed L, solve the Riccati of opt wrt K
cK = min(eig(bRw - bD'*bPK*bD))
Jinit = trace(bPKL);
cL = min(eig(bRu + bB'*bPL*bB))
% if cL < 0 & cK < 0
%     return;
% % end
% end

% solve GARE
[X, L, G] = dare(bA, [bB bD], bQ, [bRu, zeros(size(bRu)); zeros(size(bRu)) -bRw]);
Kop = (inv(bRu+bB'*(X-X*bD*inv(-bRw + bD'*X*bD)*bD'*X)*bB)*bB'*X*(bA-bD*inv(-bRw + bD'*X*bD)*bD'*X*bA)).*bKmask;
Kop2 = inv(bRu)*(bB'*X*inv(eye(18)+(bB*inv(bRu)*bB' - bD*inv(bRw)*bD')*X)*bA);
Lop = (inv(-bRw + bD'*(X-X*bB*inv(bRu + bB'*X*bB)*bB'*X)*bD)*bD'*X*(bA-bB*inv(bRu+bB'*X*bB)*bB'*X*bA)).*bLmask;
Lop2 = -inv(bRw)*(bD'*X*inv(eye(18)+(bB*inv(bRu)*bB' - bD*inv(bRw)*bD')*X)*bA);
Jop = trace(Solve_Lya(N, bA, bB, bD, bQ, bRu, bRw, Kop, Lop)*Sigma0);


%% PG-NPG
cost1 = zeros(1000000, 1);
cost1(1) = Jinit;
gradnormL1 = zeros(1000000, 1);
gradnormK1 = zeros(1000000, 1);
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
    bestL = trace(Solve_Riccati_K(N, bA, bB, bD, bQ, bRu, bRw, bK1));
    currL = trace(Solve_Lya(N, bA, bB, bD, bQ, bRu, bRw, bK1, bL1));
    gradNormL1sum = 0;
    while bestL > currL + 0.001
       bPKL = Solve_Lya(N, bA, bB, bD, bQ, bRu, bRw, bK1, bL1);
       SigmaL = SolveSigmaL(N, bA, bB, bD, bK1, bL1, Sigma0);
       PG_L = 2*((-bRw + bD'*bPKL*bD)*bL1 - bD'*bPKL*(bA-bB*bK1))*SigmaL;
       %calculate average grad norm
       gradNormL1sum = gradNormL1sum + norm(PG_L, 'fro');
       gradnormL1(idx) = gradNormL1sum/(idx+1-old_idx1);
       gradnormK1(idx) = inf;
       bL1 = bL1 + etaPG_L*PG_L;
       idx = idx + 1;
       bPKL = Solve_Lya(N, bA, bB, bD, bQ, bRu, bRw, bK1, bL1);
       cK1(idx) = min(eig(bRw - bD'*bPKL*bD));
       currL = trace(bPKL);
       cost1(idx) = currL;
    end
    gradnormL1(idx) = inf;
    gradnormK1sum = gradnormK1sum +(norm(NPG_K*SigmaL, 'fro'));
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
    trace(bPKL)
    cost1(idx) = trace(bPKL);
end

%% NPG-NPG
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
for i = 1:300000
    if mod(i, 1000) == 0
        i
    end
    idx = old_idx2;
    % calculate how well the inner loop can do
    bestL = trace(Solve_Riccati_K(N, bA, bB, bD, bQ, bRu, bRw, bK2));
    currL = trace(Solve_Lya(N, bA, bB, bD, bQ, bRu, bRw, bK2, bL2));
    gradNormL2sum = 0;
    while bestL > currL + 0.001
%     for j = 1:10
       bPKL = Solve_Lya(N, bA, bB, bD, bQ, bRu, bRw, bK2, bL2);
       SigmaL = SolveSigmaL(N, bA, bB, bD, bK2, bL2, Sigma0);
       NPG_L = 2*((-bRw + bD'*bPKL*bD)*bL2 - bD'*bPKL*(bA-bB*bK2));
       PG_L = 2*((-bRw + bD'*bPKL*bD)*bL2 - bD'*bPKL*(bA-bB*bK2))*SigmaL;
       %calculate average grad norm
       gradNormL2sum = gradNormL2sum + norm(PG_L, 'fro');
       gradnormL2(idx) = gradNormL2sum/(idx+1-old_idx2);
       gradnormK2(idx) = inf;
       bL2 = bL2 + etaNPG_L*NPG_L;
       idx = idx + 1;
       bPKL = Solve_Lya(N, bA, bB, bD, bQ, bRu, bRw, bK2, bL2);
       cK2(idx) = min(eig(bRw - bD'*bPKL*bD));
       currL = trace(bPKL);
       cost2(idx) = currL;
    end
    gradnormL2(idx) = inf;
    gradnormK2sum = gradnormK2sum +(norm(NPG_K*SigmaL, 'fro'));
    gradnormK2(idx) = gradnormK2sum/i;
    idx = idx+1;
    old_idx2 = idx;
    bPKL = Solve_Lya(N, bA, bB, bD, bQ, bRu, bRw, bK2, bL2);
    NPG_K = 2*((bRu + bB'*bPKL*bB)*bK2-bB'*bPKL*(bA-bD*bL2));
    bK2 = bK2 - alphaNPG_K2hard*NPG_K;
    %check IR
    bPK = Solve_Riccati_K(N, bA, bB, bD, bQ, bRu, bRw, bK2);
    if min(eig(bRw - bD'*bPK*bD)) < 0
%         disp('out of cK');
%         return;
    end
    bPKL = Solve_Lya(N, bA, bB, bD, bQ, bRu, bRw, bK2, bL2);
    cK2(idx) = min(eig(bRw - bD'*bPKL*bD));
    trace(bPKL)
    cost2(idx) = trace(bPKL);
end

ImageDPI=400;
linewidth = 3;
cost2size = 30000;
cK2size = 30000;
figure
hold on;
%yyaxis left
semilogy(0:cost2size-2,(cost2(1:cost2size-1)),'-','color',[0, 0.4470, 0.7410], 'linewidth',linewidth,'markersize', 1);
% ylim([0, 600])
% yticks([200, 400, 600])
ylim([0, 2500])
yticks([500, 1000, 1500, 2000, 2500])
ylabel('$\mathcal{G}(\bf{K}, \bf{L})$','Interpreter','latex', 'FontSize',22)
yyaxis right
plot(0:cK2size-2,(cK2(1:cK2size-1)),'-','color',[0.8500, 0.3250, 0.0980], 'linewidth',linewidth, 'markersize', 1);
yticks([1, 2, 3, 4, 5, 6])
ylim([0, 6])
ylabel('$\lambda_{\min}(\bf{R}^w-\bf{D}^T\bf{P}_{\bf{K}, \bf{L}}\bf{D})$','Interpreter','latex', 'FontSize',22)
legend('NPG-NPG, $\mathcal{G}$','NPG-NPG, $\lambda_{\min}$','Interpreter','latex','Location','northeast', 'FontSize',22)
xlabel('Total Iterations $K \times L$','Interpreter','latex' ,'FontSize',22)
set(gca,'FontSize', 24);

figure
hold on;
gradnormL2size = 30000;
gradnormK2size = 30000;
%yyaxis left
ylabel('Avg. $|\nabla_{\bf{L}}|_F$','Interpreter','latex', 'FontSize',22)
semilogy(0:gradnormL2size-2,(gradnormL2(1:gradnormL2size-1)),'.','color',[0.4940, 0.1840, 0.5560], 'linewidth',linewidth, 'markersize', 13);
% yticks([20, 40, 60, 80, 100, 120])
yticks([200, 400, 600])
yyaxis right
semilogy(0:gradnormK2size-2,(gradnormK2(1:gradnormK2size-1)),'.','color',[0.4660, 0.6740, 0.1880], 'linewidth',linewidth, 'markersize', 22);
legend('NPG-NPG, Avg.$|\nabla_{\bf{L}}|_F$','NPG-NPG, Avg.$|\nabla_{\bf{K}}|_F$', 'Interpreter','latex','Location','northeast', 'FontSize',22)
xlabel('Total Iterations $K \times L$','Interpreter','latex' ,'FontSize',22)
ylabel('Avg. $|\nabla_{\bf{K}}|_F$','Interpreter','latex', 'FontSize',22)
ylim([0, inf])
% yticks([1 , 2, 3, 4, 5, 6]*1e4)
yticks([1, 2, 3, 4, 5, 6]*1e6)
set(gca,'FontSize', 24);

%% GN-GN
cost3 = zeros(100000, 1);
cost3(1) = Jinit;
gradnormL3 = zeros(100000, 1);
gradnormK3 = zeros(100000, 1);
cK3 = zeros(100000, 1);
cK3(1) = min(eig(bRw - bD'*bPKL*bD));
old_idx3 = 1;
bK3 = bK;
bL3 = bL;
gradnormK3sum = 0;
for i = 1:K_iter3
    i
    idx = old_idx3;
    % calculate how well the inner loop can do
    bestL = trace(Solve_Riccati_K(N, bA, bB, bD, bQ, bRu, bRw, bK3));
    currL = trace(Solve_Lya(N, bA, bB, bD, bQ, bRu, bRw, bK3, bL3));
    gradNormL3sum = 0;
    while bestL > currL + 0.001
       bPKL = Solve_Lya(N, bA, bB, bD, bQ, bRu, bRw, bK3, bL3);
       SigmaL = SolveSigmaL(N, bA, bB, bD, bK3, bL3, Sigma0);
       NPG_L = 2*((-bRw + bD'*bPKL*bD)*bL3 - bD'*bPKL*(bA-bB*bK3));
       GN_L = inv(bRw-bD'*bPKL*bD)*NPG_L;
       PG_L = NPG_L*SigmaL;
       %calculate average grad norm
       gradNormL3sum = gradNormL3sum + norm(PG_L, 'fro');
       gradnormL3(idx) = gradNormL3sum/(idx+1-old_idx3);
       gradnormK3(idx) = inf;
       bL3 = bL3 + etaGN_L*GN_L;
       idx = idx + 1;
       bPKL = Solve_Lya(N, bA, bB, bD, bQ, bRu, bRw, bK3, bL3);
       cK3(idx) = min(eig(bRw - bD'*bPKL*bD));
       currL = trace(bPKL);
       cost3(idx) = currL; 
    end
    gradnormL3(idx) = inf;
    gradnormK3sum = gradnormK3sum +(norm(NPG_K*SigmaL, 'fro'));
    gradnormK3(idx) = gradnormK3sum/i;
    idx = idx+1;
    old_idx3 = idx;
    bPKL = Solve_Lya(N, bA, bB, bD, bQ, bRu, bRw, bK3, bL3);
    GN_K = 2*inv(bRu+bB'*bPKL*bB)*((bRu + bB'*bPKL*bB)*bK3-bB'*bPKL*(bA-bD*bL3));
    bK3 = bK3 - alphaGN_K3*GN_K;
    % check IR
    bPK = Solve_Riccati_K(N, bA, bB, bD, bQ, bRu, bRw, bK3);
    if min(eig(bRw - bD'*bPK*bD)) < 0
        disp('out of cK');
        return;
    end
    bPKL = Solve_Lya(N, bA, bB, bD, bQ, bRu, bRw, bK3, bL3);
    cK3(idx) = min(eig(bRw - bD'*bPKL*bD));
    cost3(idx) = trace(bPKL);
end

ImageDPI=400;
linewidth = 3;
cost3size = find(cost3==0, 1, 'first')
cK3size = find(cK3==0, 1, 'first')
figure
idx = 2000;
hold on;
yyaxis left
semilogy(0:idx-1,(cost3(1:idx)),'-','color',[0, 0.4470, 0.7410], 'linewidth',linewidth,'markersize', 1);
% yticks([200, 400, 600])
% ylim([0, 600])
ylim([0, 2500])
yticks([500, 1000, 1500, 2000, 2500])
ylabel('$\mathcal{G}(\bf{K}, \bf{L})$','Interpreter','latex', 'FontSize',22)
yyaxis right
plot(0:idx-1,(cK3(1:idx)),'-','color',[0.8500, 0.3250, 0.0980], 'linewidth',linewidth, 'markersize', 1);
yticks([1, 2, 3, 4, 5, 6])
ylim([0, 6])
ylabel('$\lambda_{\min}(\bf{R}^w-\bf{D}^T\bf{P}_{\bf{K}, \bf{L}}\bf{D})$','Interpreter','latex', 'FontSize',22)
legend('GN-GN, $\mathcal{G}$','GN-GN, $\lambda_{\min}$','Interpreter','latex','Location','northeast', 'FontSize',22)
xlabel('Total Iterations $K \times L$','Interpreter','latex' ,'FontSize',22)
set(gca,'FontSize', 24);

figure
hold on;
gradnormL3size = find(gradnormL3==0, 1, 'first');
gradnormK3size = find(gradnormK3==0, 1, 'first');
yyaxis left
ylabel('Avg. $\|\nabla_{\bf{L}}\|_F$','Interpreter','latex', 'FontSize',22)
semilogy(0:idx-1,(gradnormL3(1:idx)),'.','color',[0.4940, 0.1840, 0.5560], 'linewidth',linewidth, 'markersize', 15);
% yticks([20, 40, 60, 80, 100, 120])
% yticks([100, 200, 300, 400, 500, 600])
ylim([0, 600])
yticks([200, 400, 600])
yyaxis right
semilogy(0:idx-1,(gradnormK3(1:idx)),'.','color',[0.4660, 0.6740, 0.1880], 'linewidth',linewidth, 'markersize', 15);
legend('GN-GN, Avg.$\|\nabla_{\bf{L}}\|_F$','GN-GN, Avg.$\|\nabla_{\bf{K}}\|_F$', 'Interpreter','latex','Location','northeast', 'FontSize',22)
xlabel('Total Iterations $K \times L$','Interpreter','latex' ,'FontSize',22)
ylabel('Avg. $\|\nabla_{\bf{K}}\|_F$','Interpreter','latex', 'FontSize',22)
ylim([0, inf])
% yticks([1 , 2, 3, 4, 5]*1e4)
yticks([1, 2, 3, 4, 5, 6]*1e6)
set(gca,'FontSize', 24);

%% Plot Results
ImageDPI=400;
linewidth = 3;

cost1size = 600000
cK1size = 600000
figure
hold on;
yyaxis left
semilogy(0:cost1size-2,(cost1(1:cost1size-1)),'-','color',[0, 0.4470, 0.7410], 'linewidth',linewidth,'markersize', 1);
yticks([200, 400, 600])
ylim([0, 600])
ylabel('$\mathcal{G}(\bf{K}, \bf{L})$','Interpreter','latex', 'FontSize',22)
yyaxis right
plot(0:cK1size-2,(cK1(1:cK1size-1)),'-','color',[0.8500, 0.3250, 0.0980], 'linewidth',linewidth, 'markersize', 1);
yticks([1, 2, 3, 4, 5, 6])
ylim([0, 6])
ylabel('$\lambda_{\min}(\bf{R}^w-\bf{D}^T\bf{P}_{\bf{K}, \bf{L}}\bf{D})$','Interpreter','latex', 'FontSize',22)
legend('PG-NPG, $\mathcal{G}$','PG-NPG, $\lambda_{\min}$','Interpreter','latex','Location','northeast', 'FontSize',22)
xlabel('Total Iterations $K \times L$','Interpreter','latex' ,'FontSize',22)
set(gca,'FontSize', 24);



figure
hold on;
gradnormL1size = 600000;
gradnormK1size = 600000;
yyaxis left
ylabel('Avg. $|\nabla_{\bf{L}}|_F$','Interpreter','latex', 'FontSize',20)
semilogy(0:gradnormL1size-1,(gradnormL1(1:gradnormL1size)),'.','color',[0.4940, 0.1840, 0.5560], 'linewidth',linewidth, 'markersize', 13);
yticks([100, 200, 300, 400, 500, 600])
yyaxis right
semilogy(0:gradnormK1size-1,(gradnormK1(1:gradnormK1size)),'.','color',[0.4660, 0.6740, 0.1880], 'linewidth',linewidth, 'markersize', 13);
legend('PG-NPG, Avg.$|\nabla_{\bf{L}}|_F$','PG-NPG, Avg.$|\nabla_{\bf{K}}|_F$', 'Interpreter','latex','Location','northeast', 'FontSize',22)
xlabel('Total Iterations $K \times L$','Interpreter','latex' ,'FontSize',22)
ylabel('Avg. $|\nabla_{\bf{K}}|_F$','Interpreter','latex', 'FontSize',22)
ylim([0, inf])
yticks([1 , 2, 3, 4, 5, 6]*1e4)
set(gca,'FontSize', 24);

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

