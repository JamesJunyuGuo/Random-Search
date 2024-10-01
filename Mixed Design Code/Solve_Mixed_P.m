function [P, Pt] = Solve_Mixed_P(N, bA, bB, bW, bQ, bR, bK, gamma)
% Solve Lyapunov equation to find P for any given bK
P = bQ + bK'*bR*bK;
for i = 1:N
    Pt = (eye(size(P))-gamma*P*bW)\P;
%     Ptilde = P + P*bD*inv(gamma*eye(size(bR)) - bD'*P*bD)*bD'*P;
    P = bQ + bK'*bR*bK + (bA-bB*bK)'*Pt*(bA-bB*bK);
end


