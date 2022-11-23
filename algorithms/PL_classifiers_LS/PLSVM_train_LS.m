function model = PLSVM_train_LS(train_data,train_p_target,lambda,T,patience)
% A maximum margin approach to partial label learning.
% This function is the training phase of the algorithm. 
%
%    Syntax
%
%      model = PLSVM_train_LS( trainData,trainTarget,k,alpha )
%
%    Description
%
%       PLSVM_train takes,
%           train_data                  - A Q x 1 cell, the training set for the jth class label is stored in train_data{j}
%           train_p_target              - A QxM array, if the jth class label is one of the partial labels for the ith training instance, then train_p_target(j,i) equals +1, otherwise train_p_target(j,i) equals 0
%           lambda                      - The regularization parameter (defalut 1)
%           T                           - The  maximum number of iterations 
%      and returns,
%           model is a structure continues following elements
%           model.w                     - A Q x 1 cell, the parameters of the svm model for the jth class label is stored in model.w{j}.
%           model.label_num             - # labels of training data


%  [1]N. Nguyen and R. Caruana, “Classification with partial labels,”in Proceedings of the 14th ACM SIGKDD International Conference onKnowledge Discovery and Data Mining, Las Vegas, NV, 2008, pp.381–389.
    
    if nargin<5
        patience = 10;
    end
    if nargin<4
        T = 1000;
    end
    if nargin<3
        lambda = 1;
    end
    if nargin<2
        error('Not enough input parameters, please check again.');
    end


    label_num = size(train_p_target,1); 
    
    w = cell(label_num, 1);
    w_v = [];
    for j=1:label_num
        w{j} = rand(size(train_data{j}, 2), 1);
        w_v = [w_v; w{j}];
    end
    normw = norm(w_v);
    if normw>1/lambda^0.5
        for j=1:label_num
            w{j} = w{j}/(lambda^0.5*normw);
        end
    end
    
    ins_num = size(train_data{1},1);
    
    loss_old = Inf; 
    patience_left = patience;
    insList = 1:ins_num;
    for t=1:T
        if mod(t,10)==0
            disp(['PL_SVM iteration: ',num2str(t)]);
        end
        
        value = zeros(label_num, ins_num);
        for j = 1:label_num
            value(j, :) = w{j}'*train_data{j}';
        end
        [maxPosVal,maxPosIdx] = max(value-1e5*(train_p_target==0));
        [maxNegVal,maxNegIdx] = max(value-1e5*(train_p_target==1));

        violateIdx = insList((maxPosVal-maxNegVal)<1);
        violateLabelPos = maxPosIdx(violateIdx); 
        violateLabelNeg = maxNegIdx(violateIdx);
        wtemp = cell(label_num, 1);
        wtemp_v = [];
        niu = 1/(lambda*t);
        for label=1:label_num
            
            violate_label = violateIdx(violateLabelPos==label);
            wtemp{label} = sum(train_data{label}(violate_label,:));
            violate_label = violateIdx(violateLabelNeg==label);
            wtemp{label} = wtemp{label} - sum(train_data{label}(violate_label,:)); %gd
            wtemp{label} = wtemp{label}'*niu/ins_num+(1-niu*lambda)*w{label};
            wtemp_v = [wtemp_v; wtemp{label}];
        end
        
        normw = norm(wtemp_v);
        if 1/(lambda^0.5*normw)<1
            for j=1:label_num
                w{j} = wtemp{j}*(1/(lambda^0.5*normw));
            end
        else
            w = wtemp;
        end
        xi = 1-maxPosVal+maxNegVal;
        xi(xi<0) = 0;
        
        w_v = [];
        for j=1:label_num
            w_v = [w_v; w{j}];
        end
        normw = norm(w_v);
        loss = normw*lambda/2+mean(xi);
        fprintf('Loss %.4f\n', loss);
        if abs(loss_old-loss)<=1e-4
            break
        end
        loss_old = loss;

    end
    model.w = w;
    model.type = 'PLSVM';
    model.label_num = label_num;
end

