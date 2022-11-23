function Y = LabelProp(S, partial_y, alpha, k)
%    Syntax
%
%       Y = LabelProp(S, partial_y, alpha)
%
%    Description
%
%       LabelProp takes,
%           S                  - An MxM array, similarity matrix among data
%           partial_y          - An MxQ array, if the jth class label is one of the partial labels for the ith instance, then partial_y(i,j) equals +1, otherwise partial_y(i,j) equals 0
%           alpha              - A balancing coefficient parameter which 0<alpha<1 (defalut 0.95)
%           k                  - The number of nearest neighbors be considered     (defalut 10)
% 
%      and returns,
%           Y                  - An MxQ array, the labeling confidence matrix
    
    if nargin<4
        k = 10;
    end
    if nargin<3
        alpha = 0.95;
    end
    
    % truncate the similarity matrix by keeping the elements in nearest neighbors
    num_data = size(S, 1);
    S = S - diag(diag(S));
    S_trunc = zeros(size(S));
    [~, inds] = maxk(S, k, 2);
    for i = 1:num_data
        S_trunc(i, inds(i, :)) = S(i, inds(i, :));
    end
    S = S_trunc;
    
    % initial label confidence matrix
    Y0 = partial_y ./ sum(partial_y, 2);
    
    % normalized similarity matrix
    D = sum(S, 2);
    D(D==0)=1;
    P = S./D;
    
    % update label confidence matrix via label propagation
    Y = Y0;
    for iter=1:100
        old_Y = Y;
        Y = alpha * P * Y + (1-alpha) * Y0;
        Y = Y .* partial_y;
        Y = Y ./ sum(Y, 2);
        diff = norm(old_Y-Y, 2);
        if abs(diff)<0.0001
            break
        end
    end
end