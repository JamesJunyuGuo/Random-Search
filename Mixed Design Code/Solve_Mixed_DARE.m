function [Ps, Pts] = Solve_Mixed_DARE(N, bA, bB, bW, bQ, bR, gamma)
% Solve Riccati equation to find Ps for any given bK
Ps = bQ;
for i = 1:N
    Pts = (eye(size(Ps))-gamma*Ps*bW)\Ps;
    bK = (bR+bB'*Pts*bB)\(bB'*Pts*bA);
    Ps = bQ + bK'*bR*bK + (bA-bB*bK)'*Pts*(bA-bB*bK);
end

