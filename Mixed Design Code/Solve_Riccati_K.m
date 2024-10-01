function bPK = Solve_Riccati_K(N,bA, bB, bD, bQ, bRu, bRw, bK)
% For a fixed K, solve Riccati for inner loop L
bPK = bQ + bK'*bRu*bK;
for i = 1:N
    bPK = bQ + bK'*bRu*bK + (bA-bB*bK)'*(bPK + bPK*bD*pinv(bRw-bD'*bPK*bD)*bD'*bPK)*(bA-bB*bK);
end


