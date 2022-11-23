function [P] = solveQP(Y, Q, H, Aeq, beq, lb, opts, lambda)
%% solve m QP problems
[m, l] = size(Q);
P = zeros(m, l);

for i = 1:m
    ub = Y(i,:)';
    indicesOfCandidates = find(Y(i,:)==1);
    [~, index] = max(Q(i,indicesOfCandidates));
    j = indicesOfCandidates(index);
    atp = zeros(l,1);
    atp(j,1) = 1;
    f = -2*Q(i,:)' - lambda*atp;
    Amtp = repmat(atp', l, 1);
    Aneq = eye(l,l) - Amtp; % I*p - Amtp*p <= 0
    bneq = zeros(l,1);
    
    x = quadprog(H, f, Aneq, bneq, Aeq, beq, lb, ub, [], opts);

    P(i,:) = x';
end

end

%%