clear; clc;
addpath(genpath('tools'));
addpath(genpath('algorithms'));
addpath(genpath('libsvm-3.24/libsvm-3.24'));
dataset_dir = 'Datasets/';
dataset_loader = DatasetLoader(dataset_dir);
%% Setting parameters for evaluation
% setting datasets on which algorithms are evaluated
real_world_dataset_names = {'lost', 'MSRCv2', 'BirdSong', 'Soccer Player', 'Mirflickr'};
dataset_names = {'glass'};% input datasets: synthetic datasets ('glass',...) or real wolrd datasets ('lost',...) 
% setting paramters for cross-validation
nfold = 10;
run_all_folds = true; % if true, perferm 10-fold cross-validation; otherwise, train and validate on the first fold
mode = 'co-occur'; % mode can be set as 'random', 'co-occur' or 'real-world', corresponding to three different experimental setups
p = 1; % ratio of partial label examples, can be set as [0.1, 0.2, ..., 0.7, 1.0]. p is valid when mode is 'random' or 'co-occur'
num_FL = 1; % (r) number of false positive labels, can be set as [1, 2, 3]. num_FL is valid when mode is 'random'
epsilon = 0.5; % probability for picking co-occurring label as false postive label, can be set as [0.1, 0.2, ..., 0.7]. epsilon is valid when mode is 'co-occur'

% setting dir for saving log and learned model
exp_dir = 'Exps_UCL_SVM';
if exist(exp_dir, 'dir') == 0
    mkdir(exp_dir)
end
log_dir = [exp_dir, '/Log/'];
if exist(log_dir, 'dir') == 0
    mkdir(log_dir)
end
res_dir = [exp_dir, '/res/'];
if exist(res_dir, 'dir') == 0
    mkdir(res_dir)
end

% creating log file
save_file = true;
file_name = [log_dir, datestr(now, 'yyyy-mm-dd_HH-MM-SS'), '.txt'];
fileID = fopen(file_name, 'w');
fprintf(fileID, '%d-fold cross-validation experiment\n\n', nfold);
% parameters for PL-LIFT
r = 0.2; % [0.1, 0.2, 0.3, 0.4, 0.5]
tl = 0.2; % [0.1, 0.2, 0.3, 0.4]
th = 0.6; % [0.5, 0.6, 0.7, 0.8, 0.9]
alpha = 0.5; % [0.1, 0.2, ..., 0.9, 1.0]
preprocess_before_PLLIFT = true;
preprocess_before_cls = true;

%% Evaluating SVM_LIFT_Spectral_3P algorithm
% setting paramters for algorithm
algorithm_name = 'SVM_LIFT_Spectral_3P';

for dataset_name = dataset_names
    dataset_name_str = dataset_name{1};
    fprintf('Evluating %s on %s dataset\n', algorithm_name, dataset_name_str);
    fprintf('r=%.1f, tl=%.1f, th=%.1f, alpha=%.1f\n', r, tl, th, alpha);
    if (save_file)
        fprintf(fileID, '\nEvluating %s on %s dataset\n', algorithm_name, dataset_name_str);
        fprintf(fileID, 'r=%.1f, tl=%.1f, th=%.1f, alpha=%.1f\n', r, tl, th, alpha);
        fprintf(fileID, 'prepro1=%d, prepro2=%d\n', ...
             preprocess_before_PLLIFT, preprocess_before_cls);
        fprintf(fileID, '\nmode=%s\n', mode);
        if strcmp(mode,'random')
             fprintf(fileID, '\np=%d,num_FL=%d\n', p,num_FL);
        elseif strcmp(mode,'co-occur')
            fprintf(fileID, '\np=%d,epsilon=%.1f\n', p,epsilon);
        end
    end
    
    % loading dataset
    if ismember(dataset_name_str, real_world_dataset_names)
        mode = 'real-world';
    end
    if preprocess_before_PLLIFT
        dataset = dataset_loader.Load(dataset_name_str, 3, mode);
    else
        dataset = dataset_loader.Load(dataset_name_str, 0, mode);
    end

 
    % cross-validation
    accuracy_list = zeros(nfold, 1);
    train_times = zeros(nfold, 1);
    test_times = zeros(nfold, 1);
    for i = 1:nfold
        fprintf('cross-validation: [%d / %d]\n', i, nfold);
        
        % splitting the dataset
        [train_index, test_index] = dataset.Partition(i, nfold);
        [train_data, train_partial_y, val_data, val_y] = dataset.Get_set(train_index, test_index, mode, p, num_FL, epsilon);
        
        tic;
        [centers, imps, new_train_data, Y] = PL_LIFT_Spectral_3P(train_data, train_partial_y, r, tl, th, alpha);
        
        % transformation for validation set
        num_classes = size(centers, 1);
        new_val_data = cell(size(centers));
        for j = 1:num_classes
            val_data = val_data./vecnorm(val_data, 2, 2);%
            new_val_data{j} = pdist2(val_data, centers{j}) .* imps{j}';
        end

        if preprocess_before_cls
            for j = 1:num_classes
                num_train = size(new_train_data{j}, 1);
                norm_data = zscore([new_train_data{j}; new_val_data{j}]);
                new_train_data{j} = norm_data(1:num_train, :);
                new_val_data{j} = norm_data(num_train+1:end, :);
            end
        end
       
        % training and validating
        rng(5); % random seed
     
        model = PLSVM_train_LS(new_train_data,train_partial_y'); 
        train_times(i) = toc;
        tic;
        [accuracy,predictLabel,outputValue] = PLSVM_predict_LS(new_val_data,val_y',model);
        test_times(i) = toc;
        accuracy_list(i) = accuracy;
        fprintf('Accuracy=%.4f\n', accuracy);
        if (save_file)
            fprintf(fileID, 'cross-validation [%d / %d]: Accuracy=%.4f\n', i, nfold, accuracy);
            fprintf(fileID, 'Training time for %d fold: %.3f s\n', i, train_times(i));
            fprintf(fileID, 'Test time for %d fold: %.5f s\n', i, test_times(i));
            if preprocess_before_cls
                res_file_name = sprintf('%s_zscore_cv%d_res.mat', dataset.dataset_name, i);
            else
                res_file_name = sprintf('%s_cv%d_res.mat', dataset.dataset_name, i);
            end
            file_name = [res_dir, res_file_name];
            save(file_name, 'predictLabel', 'outputValue', 'model'); % save predictions and models
        end
        
        if run_all_folds == false
           break 
        end
    end
    
    if run_all_folds
        mean_acc = mean(accuracy_list);
        std_acc = std(accuracy_list);
        fprintf('Accuracy: %.4f / %.4f\n', mean_acc, std_acc);
        if (save_file)
            fprintf(fileID, 'Accuracy: %.4f / %.4f\n', mean_acc, std_acc);
            fprintf(fileID, 'Average training time: %.3f s\n', mean(train_times));
            fprintf(fileID, 'Average test time: %.5f s\n\n', mean(test_times));
        end
    end
end
fprintf(fileID, '\n\n');
%%
fclose(fileID);