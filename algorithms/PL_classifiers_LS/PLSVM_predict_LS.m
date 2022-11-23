function [accuarcy,predictLabel,outputValue] = PLSVM_predict_LS(test_data,test_target,model)
% A maximum margin approach to partial label learning.
% This function is the prediction phase of the algorithm. 
%    Syntax
%
% [accuarcy,predictLabel,outputValue] = PLSVM_predict_LS(test_data,test_target,model)
% 
%    Description
%
%       CLPL_predict takes,
%           model                       - the model which returned in the training phase
%           test_data                   - A Q x 1 cell, the training set for the jth class label is stored in test_data{j}
%           test_target                 - A QxM array, if the jth class label is the ground-truth label for the ith test instance, then test_target(j,i) equals 1; otherwise test_target(j,i) equals 0
% 
%      and returns,
%            accuarcy                   - Predictive accuracy on the test set
%            predictLabel               - A QxM array, if the ith test instance is predicted to have the jth class label, then predictLabel(j,i) is 1, otherwise predictLabel(j,i) is 0
%            outputValue                - A QxM array, the numerical output of the ith test instance on the jth class label is stored in Outputs(j,i)
%  [1]N. Nguyen and R. Caruana, “Classification with partial labels,”in Proceedings of the 14th ACM SIGKDD International Conference onKnowledge Discovery and Data Mining, Las Vegas, NV, 2008, pp.381–389.

    if nargin<3
        error('Not enough input parameters, please check again.');
    end

    if strcmp(model.type,'PLSVM')==0
        error('The input model does not match the prediction model')
    end
    
    q = model.label_num;
    ins_num = size(test_data{1}, 1);
    outputValue = zeros(q, ins_num);
    for j = 1:q
        outputValue(j, :) = model.w{j}'*test_data{j}';
    end
    [~,predictLabel] = max(outputValue);
    [~,real] = max(full(test_target));
    [label_num,test_num] = size(test_target);
    accuarcy = sum(predictLabel==real)/size(test_data{1},1);
        LabelMat = repmat((1:label_num)',1,test_num);
    predictLabel = repmat(predictLabel,label_num,1)==LabelMat;