function SigmaL = SolveSigmaL(N, bA, bB, bD, bK, bL, Sigma0)
% Solve the Lyapunov equation to obtain SigmaK
SigmaL = Sigma0;
for i = 1:N
    SigmaL = Sigma0 + (bA-bB*bK-bD*bL)*SigmaL*(bA-bB*bK-bD*bL)';
end



