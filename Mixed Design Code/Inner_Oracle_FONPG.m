function Est_bL = Inner_Oracle_FONPG(bK, bL, N, bA, bB, bD, bQ, bRu, bRw, Sigma0, eps1, etaNPG_L)
    % for a fixed K and accuracy level eps1, return \hat{L}(K) that
    % achieves eps1 optimum
    bestL = trace(Solve_Riccati_K(N, bA, bB, bD, bQ, bRu, bRw, bK)*Sigma0);
    currL = trace(Solve_Lya(N, bA, bB, bD, bQ, bRu, bRw, bK, bL)*Sigma0);
    while bestL > currL + eps1
       bPKL = Solve_Lya(N, bA, bB, bD, bQ, bRu, bRw, bK, bL);
       NPG_L = 2*((-bRw + bD'*bPKL*bD)*bL - bD'*bPKL*(bA-bB*bK));
%        SigmaL = SolveSigmaL(N, bA, bB, bD, bK, bL, Sigma0);
       %NPG
       bL = bL + etaNPG_L*NPG_L;
       
       
       currL = trace(Solve_Lya(N, bA, bB, bD, bQ, bRu, bRw, bK, bL)*Sigma0);
    end
    Est_bL = bL;
end