function Est_bL = Inner_Oracle_ZONPG(bK, bL, bLmask, bA, bB, bD, bQ, bRu, bRw, Sigma0, M1, m, n, N, r1, var, eps1, etaNPG_L)
    % for a fixed K and accuracy level eps1, return \hat{L}(K) that
    % achieves eps1 optimum
    bestL = trace(Solve_Riccati_K(N, bA, bB, bD, bQ, bRu, bRw, bK)*Sigma0);
    currL = trace(Solve_Lya(N, bA, bB, bD, bQ, bRu, bRw, bK, bL)*Sigma0);
%     gradNormL1sum = 0;
    while bestL > currL + eps1
%        bestL - currL
%        bPKL = Solve_Lya(N, bA, bB, bD, bQ, bRu, bRw, bK1, bL1);
%        Sigma_L = SolveSigmaL(N, bA, bB, bD, bK1, bL1, Sigma0);
%        PG_L = 2*((-bRw + bD'*bPKL*bD)*bL1 - bD'*bPKL*(bA-bB*bK1))*Sigma_L;
%        bL1;
       [Est_PG_L, Est_Sigma_L] = InnerGradEst(bK, bL, bLmask, bA, bB, bD, bQ, bRu, bRw, Sigma0, M1, m, n, N, r1, var);
       %calculate average grad norm
%        norm(PG_L-Est_PG_L)
       Est_NPG_L = Est_PG_L/Est_Sigma_L;
%        gradNormL1sum = gradNormL1sum + norm(Est_NPG_L, 'fro');
%        gradnormL1(idx) = gradNormL1sum/(idx+1-old_idx1);
%        gradnormK1(idx) = inf;
       bL = bL + etaNPG_L*Est_NPG_L;
%        idx = idx + 1;
%        bPKL = Solve_Lya(N, bA, bB, bD, bQ, bRu, bRw, bK, bL);
%        cK1(idx) = min(eig(bRw - bD'*bPKL*bD));
       currL = trace(Solve_Lya(N, bA, bB, bD, bQ, bRu, bRw, bK, bL)*Sigma0);
%        cost1(idx) = currL;
    end
    Est_bL = bL;
end