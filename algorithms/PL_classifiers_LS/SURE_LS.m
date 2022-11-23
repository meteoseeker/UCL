function [accuarcy,predictLabel,outputValue,train_time,test_times] = SURE_LS(train_data, train_p_target, test_data, test_target, optmParameter)

max_iter = optmParameter.maxIter;
lambda = optmParameter.lambda;
beta = optmParameter.beta;

tic;
ker = 'rbf';
% ker = 'lin';

[num_data, num_class] = size(train_p_target);
Aeq = ones(1, num_class);
beq = 1;
opts = optimoptions('quadprog',...
    'Algorithm','interior-point-convex','Display','off');
lb = sparse(num_class, 1);
H = 2*speye(num_class, num_class);
P = train_p_target;
m1 = ones(num_data,1); I = eye(num_data, num_data);
Q = zeros(num_data, num_class);

% precompute kernel matrices
K = cell(num_class, 1);
Kt = cell(num_class, 1);
for j = 1:num_class
    par = mean(pdist(train_data{j}));
    K{j} = kernelmatrix(ker,train_data{j}',train_data{j}',par); % m by m, kernel matrix
    Kt{j} = kernelmatrix(ker,test_data{j}',train_data{j}',par); 
end

A = zeros(num_data, num_class); b = zeros(num_class, 1);
for iter = 1:max_iter
    for j = 1:num_class
        A(:, j) = (K{j} + beta*I - sum(K{j})/num_data)\(P(:, j) - sum(P(:, j))/num_data);
        b(j) = (sum(P(:, j)) - A(:, j)'*sum(K{j})')/num_data;
        Q(:, j) = K{j} * A(:, j) + b(j);
    end
    [P] = solveQP(train_p_target, Q, H, Aeq, beq, lb, opts, lambda);
end
train_time = toc;

tic;
Ytest = zeros(size(test_target));
for j = 1:num_class
    Ytest(:, j) = Kt{j} * A(:, j) + b(j);
end
outputValue = Ytest';
[~,predictLabel] = max(outputValue);
[~,real] = max(full(test_target'));
[label_num,test_num] = size(test_target');
accuarcy = sum(predictLabel==real)/test_num;
LabelMat = repmat((1:label_num)',1,test_num);
predictLabel = repmat(predictLabel,label_num,1)==LabelMat;
test_times = toc;
end