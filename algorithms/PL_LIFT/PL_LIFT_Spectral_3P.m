function [centers, imps, new_train_data, Y] = PL_LIFT_Spectral_3P(train_data, train_partial_y, r, tl, th, alpha)
% PL_LIFT_Spectral_3P 
%
% Syntax
%
%       [centers, new_train_data] = PL_LIFT_Spectral_3P(train_data, train_partial_y, neighbors, r, T, k)
%
% Description
%
%       PL_LIFT_Spectral_3P takes,
%           train_data          - An M x D array, the ith instance of training instance is stored in train_data(i,:)
%           train_partial_y     - An M x Q array, if the jth class label is one of the partial labels for the ith training instance, then train_p_target(i,j) equals +1, otherwise train_p_target(i,j) equals 0
%           r                   - Ratio parameter to control the number of clusters 
%           tl                  - Lower threshold for partitioning set of negative instances. 0 < tl < 0.5
%           th                  - Higher threshold for partitioning set of positive instances. 0.5 <= th < 1
%           alpha               - Weight for clusters of uncertain instances. 0 <= alpha <= 1
%      
%       and returns,
%           centers             - A Q x 1 cell, clusters for the jth class label is stored in clusters{j}
%           imps                - A Q x 1 cell, weights for clusters of the jth class label is stored in imps{j}
%           new_train_data      - A Q x 1 cell, new training set in the label-specific feature space, while the training set for the jth class label is stored in new_train_data{j}
%           Y                   - An M x Q array, the labeling confidence matrix
    if nargin<8
        alpha = 0.5;
    end
    if nargin<7
        th = 0.6;
    end
    if nargin<6
        tl = 0.2;
    end
    if nargin<5
        neighbors_update = 8;
    end
    if nargin<4
        T = 1;
    end
    if nargin<3
        r = 0.1;
    end
    [num_data, num_class] = size(train_partial_y);

    % precompute feature similarity
    train_data = train_data./vecnorm(train_data, 2, 2);%
    dist_temp = pdist(train_data);
    dist = squareform(dist_temp);
    sigma = mean(dist_temp);
    S = exp(-dist.^2 / (2*sigma^2));
    
    % initialize the labeling confidence matrix Y (num_data x num_class)
    Y = LabelProp(S, train_partial_y);
    
    centers = cell(num_class, 1); imps = cell(num_class, 1);
    new_train_data = cell(num_class, 1);
    B = zeros(size(Y)); % intermediate matrix (num_data x num_class)

    % perform clustering analysis for each class label
    for j = 1:num_class
        % label similarity
        neg_inds = Y(:, j) < tl;
        neg_data = train_data(neg_inds, :);
        num_neg = size(neg_data, 1);
        pos_inds = Y(:, j) > th;
        pos_data = train_data(pos_inds, :);
        num_pos = size(pos_data, 1);
        uncertain_inds = ~(neg_inds | pos_inds);
        uncertain_data = train_data(uncertain_inds, :);
        num_unc = num_data - num_neg - num_pos;
        
        % decide number of clusters
        k = ceil(r * min(num_pos, num_neg));
        k_unc = ceil(r * num_unc);
        
        % spectral cluster
        if (k == 0)
            if (num_pos == 0)
                pos_centers = [];
                neg_centers = cluster_spectral(S(neg_inds, neg_inds), min(50, num_neg), neg_data);
            else
                pos_centers = cluster_spectral(S(pos_inds, pos_inds), min(50, num_pos), pos_data);
                neg_centers = [];
            end
        elseif (k == 1)
            if (num_pos == 1)
                pos_centers = pos_data;
            else
                pos_centers = mean(pos_data);
            end
            if (num_neg == 1)
                neg_centers = neg_data;
            else
                neg_centers = mean(neg_data);
            end
        else
            % clustering on positive examples for the jth class label
            pos_centers = cluster_spectral(S(pos_inds, pos_inds), k, pos_data);
            % clustering on negative examples for the jth class label
            neg_centers = cluster_spectral(S(neg_inds, neg_inds), k, neg_data);
        end
        if (k_unc == 0)
            unc_centers = [];
        elseif (k_unc == 1)
            if (num_unc == 1)
                unc_centers = uncertain_data;
            else
                unc_centers = mean(uncertain_data);
            end
        else
            unc_centers = cluster_spectral(S(uncertain_inds, uncertain_inds), k_unc, uncertain_data);
        end
        centers{j} = [pos_centers; neg_centers; unc_centers];
        imps{j} = [ones(size(pos_centers, 1), 1); ones(size(neg_centers, 1), 1); alpha*ones(size(unc_centers, 1), 1)];
        % construct the label-specific features for the jth class label
        new_train_data{j} = pdist2(train_data, centers{j}) .* imps{j}';
            
    end  
end

function centers = cluster_spectral(S, k, data)
    centers = zeros(k, size(data, 2));
    idx = spectralcluster2(S, k, 'Distance', 'precomputed');
    for i = 1:k
        centers(i, :) = mean(data(idx==i, :));
    end
end