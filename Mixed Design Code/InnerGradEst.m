function [GradL, CovL] = InnerGradEst(bK, bL, bLmask, bA, bB, bD, bQ, bRu, bRw, Sigma0, M1, m, n, N, r1, var)
    GradL = zeros(size(bL)); CovL = zeros(size(Sigma0));
    parfor i = 1:M1 
        % random vector uniformly drawn with Fro norm = 1
%         U = (rand(size(bL))-0.5).*bLmask;
        ranMtx = normrnd(0, 1, [n*m, N]);
        ranMtx = ranMtx/norm(ranMtx, 'fro');
        U = zeros(size(bL));
        for tmp = 0:N-1
            U(tmp*n+1:(tmp+1)*n, tmp*m+1:(tmp+1)*m) = reshape(ranMtx(:,tmp+1), n, m)
        end
        % perturbed feedback gain using one point
        bL_u = bL + r1*U;
      
        % sample an initial state, input of normrnd function is standard deviation sigma.
        x_0 = normrnd(0, sqrt(var), [1,m])';
        
        % simulate one trajectory and obtain the additive cost
        Sigma0_u = zeros(size(Sigma0));
        Sigma0_u(1:m, 1:m) = x_0*x_0';
        for t = 0:N-1
           w_t = normrnd(0, sqrt(var),[1,m])';
           Sigma0_u((t+1)*m+1:(t+2)*m,(t+1)*m+1:(t+2)*m) = w_t*w_t';
        end 
        SigmaL_u = SolveSigmaL(N, bA, bB, bD, bK, bL_u, Sigma0_u);
        SigmaL_0 = SolveSigmaL(N, bA, bB, bD, bK, bL, Sigma0_u);
        J_u = trace((bQ + (bK)'*bRu*bK - (bL_u)'*bRw*bL_u) * SigmaL_u); 
        GradL = GradL + (J_u * U);
        CovL = CovL + SigmaL_0;
    end
    GradL = (m*n*N)/(M1*r1) * GradL;
    CovL = CovL/M1;
end