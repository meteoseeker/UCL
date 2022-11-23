function [accuracy,predictLabel,outputValue] = PL_KNN_LS(train_data,train_p_target,test_data,test_target,k)
%PL_KNN_LS: A k-nearest neighbor approach to partial label learning based on label-specific features
%
%    Syntax
%
%       [accuracy,predictLabel,outputValue] = PL_KNN_LS(train_data,train_p_target,test_data,test_target,k)
%
%    Description
%
%       PL_KNN_LS takes,
%           train_data     - A Q x 1 cell, the training set for the jth class label is stored in train_data{j}
%           train_p_target - A QxM array, if the jth class label is one of the partial labels for the ith training instance, then train_p_target(j,i) equals +1, otherwise train_p_target(j,i) equals 0
%           test_data      - A Q x 1 cell, the training set for the jth class label is stored in test_data{j}
%           test_target    - A QxM2 array, if the jth class label is the ground-truth label for the ith test instance, then test_target(j,i) equals 1; otherwise test_target(j,i) equals 0
%           k              - the number of the neighboors
%      and returns,
%           Outputs        - A QxM2 array, the numerical output of the ith test instance on the jth class label is stored in Outputs(j,i)
%           Pre_Labels     - A QxM2 array, if the ith test instance is predicted to have the jth class label, then Pre_Labels(j,i) is 1, otherwise Pre_Labels(j,i) is 0
%           Accuracy       - Predictive accuracy on the test set
%
    if nargin<5
        k = 10;
    end
    if nargin<2
        error('Not enough input parameters, please check again.');
    end

    [num_classes, num_test] = size(test_target);
    neighbor_list = cell(num_classes, 1);
    dis_list = cell(num_classes, 1);
    for j=1:num_classes
        kdtree = KDTreeSearcher(train_data{j}); 
        [neighbor,dis] = knnsearch(kdtree, test_data{j}, 'k', k); 
        neighbor_list{j} = neighbor;
        dis_list{j} = dis;
    end

    weighted_votes = zeros(num_test, num_classes);
    for j=1:num_classes
        dis = dis_list{j};
        neighbor = neighbor_list{j};
        
        weights = 1 - dis(:, 1:k) ./ sum(dis, 2);
        votes = train_p_target(j, neighbor(:, 1:k));
        votes = reshape(votes, [num_test, k]);
        weighted_votes(:, j) = sum(weights.*votes, 2);
    end
    [~, idx] = max(weighted_votes, [], 2);
    predictLabel = idx';
    outputValue = weighted_votes';
    
%     for test=1:num_test
%         label = zeros(1,num_classes); 
%         for j=1:num_classes
%             dis = dis_list{j};
%             neighbor = neighbor_list{j};
%             sumDis = sum(dis(test, :)); %将dis的第test列的每行元素相加
%             
%             weights = 1 - dis(test, 1:k) / sumDis;
%             votes = train_p_target(j, neighbor(test, 1:k));
%             label(1,j) = sum(weights.*votes);
%         end
%         [~,idx]=max(label);%max还返回 A 中最大值在运算维度上的对应索引。
%         predictLabel(test) = idx;
%         outputValue(:,test) = label';
%     end

    [~, real] = max(full(test_target));
    accuracy = sum(predictLabel==real)/size(test_target,2);
    LabelMat = repmat((1:num_classes)',1,num_test); %repmat复制函数，将(1:label_num)'复制1*test_num
    predictLabel = repmat(predictLabel,num_classes,1)==LabelMat;
end

