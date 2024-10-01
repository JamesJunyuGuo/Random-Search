function bPKL = Solve_Lya(N,bA, bB, bD, bQ, bRu, bRw, bK, bL)
% Solve Lyapunov equation to find PKL for any given bK, bL
bPKL = bQ + bK'*bRu*bK - bL'*bRw*bL;
for i = 1:N
    bPKL = bQ + bK'*bRu*bK - bL'*bRw*bL + (bA-bB*bK-bD*bL)'*bPKL*(bA-bB*bK-bD*bL);
end


