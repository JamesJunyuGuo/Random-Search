function bPL = Solve_Riccati_L(N,bA, bB, bD, bQ, bRu, bRw, bL)
% For a fixed K, solve Riccati for inner loop L
bPL = bQ - bL'*bRw*bL;
for i = 1:N
    bPL = bQ - bL'*bRw*bL + (bA-bD*bL)'*(bPL - bPL*bB*inv(bRu+bB'*bPL*bB)*bB'*bPL)*(bA-bD*bL);
end


