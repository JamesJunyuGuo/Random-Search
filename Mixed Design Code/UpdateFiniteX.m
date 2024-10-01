function newX = UpdateFiniteX(x0, m, d, N, bA, bB, bK, w)
    newX = zeros(m*(N+1), 1);
    newX(1:m, 1) = x0;
    for i = 1:N
        rs = i*m + 1;
        re = (i+1)*m;
        A = bA(rs:re, rs-m:re-m);
        B = bB(rs:re, (i-1)*d+1:i*d);
        K = bK((i-1)*d+1:i*d, rs-m:re-m);
        newX(rs:re, 1) = (A-B*K)*newX(rs-m:re-m, 1) + w(rs:re, 1);
    end
end

