function [GradK, CovK] = OuterGradEst(bK, bL, bKmask, bLmask, bA, bB, bD, bQ, bRu, bRw, Sigma0, M1, M2, m, n, d, N, r1, r2, var, eps1, etaNPG_L)
    GradK = zeros(size(bK)); CovK = zeros(size(Sigma0));
    parfor j = 1:M2 
        % random vector uniformly drawn with Fro norm = 1
%         V = (rand(size(bK))-0.5).*bKmask;
%         V = V/norm(V, 'fro');
        ranMtx = normrnd(0, 1, [d*m, N]);
        ranMtx = ranMtx/norm(ranMtx, 'fro');
        V = zeros(size(bK));
        for tmp = 0:N-1
            V(tmp*d+1:(tmp+1)*d, tmp*m+1:(tmp+1)*m) = reshape(ranMtx(:,tmp+1), d, m);
        end
        % perturbed feedback gain using one point
        bK_v = bK + r2*V; 

        % sample an initial state, input of normrnd function is standard deviation sigma.
        x_0 = normrnd(0, sqrt(var), [1,m])';
        
        % simulate one trajectory and obtain the additive cost
        Sigma0_v = zeros(size(Sigma0));
        Sigma0_v(1:m, 1:m) = x_0*x_0';
        for t = 0:N-1
           w_t = normrnd(0, sqrt(var),[1,m])';
           Sigma0_v((t+1)*m+1:(t+2)*m,(t+1)*m+1:(t+2)*m) = w_t*w_t';
        end 
        
        % find the approximate best-responding \hat{\bL}(\bK_v) corresponding to bK_v
        % ZO
%       Est_bL_v = Inner_Oracle_ZONPG(bK_v, bL, bLmask, bA, bB, bD, bQ, bRu, bRw, Sigma0, M1, m, n, N, r1, var, eps1, etaNPG_L);

        % FO
%         Est_bL_v = Inner_Oracle_FONPG(bK_v, bL, N, bA, bB, bD, bQ, bRu, bRw, Sigma0, eps1, etaNPG_L);
%         Est_bL_0 = Inner_Oracle_FONPG(bK, bL, N, bA, bB, bD, bQ, bRu, bRw, Sigma0, eps1, etaNPG_L);

        % Riccati
        bPK_v = Solve_Riccati_K(N, bA, bB, bD, bQ, bRu, bRw, bK_v)
        Est_bL_v = (-bRw + bD'*bPK_v*bD)\(bD'*bPK_v*(bA-bB*bK_v));
        
        % Unperturbed Riccati
        bPK = Solve_Riccati_K(N, bA, bB, bD, bQ, bRu, bRw, bK)
        Est_bL_0 = (-bRw + bD'*bPK*bD)\(bD'*bPK*(bA-bB*bK));
        
        % calculate sigmaK_v, which is the evolution of random noises and init state 
        SigmaK_v_r = SolveSigmaL(N, bA, bB, bD, bK_v, Est_bL_v, Sigma0_v);
        SigmaK_v_0 = SolveSigmaL(N, bA, bB, bD, bK, Est_bL_0, Sigma0_v);
        
        J_v = trace((bQ + bK_v'*bRu*bK_v - Est_bL_v'*bRw*Est_bL_v) * SigmaK_v_r);
        GradK = GradK + (J_v * V);
        CovK = CovK + SigmaK_v_0;
    end
    GradK = (m*d*N)/(M2*r2) * GradK;
    CovK = CovK/M2;
end