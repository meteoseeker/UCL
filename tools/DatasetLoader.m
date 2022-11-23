classdef DatasetLoader
    %DatasetLoader a dataset loader

    properties
        dataset_dir         % Dir of the dataset file
        dir
        dataset_name        % Name of the dataset
        partition_file_name % Name of the dataset partition file
        X                   % An MxN array, the ith instance is stored in X(i, :)
        y                   % An MxQ array, if the ith instance belongs to the jth class, then y(i, j) equals to +1, otherwise y(i, j) equals to 0
        partial_y           % An MxQ array, if the jth class is in the condidate label set of the ith instance, then partial_y(i, j) equals to +1, otherwise partial_y(i, j) equals to 0
    end
    
    methods
        function obj = DatasetLoader(dataset_dir)
            %DatasetLoader constructor
            
            obj.dataset_dir = dataset_dir;
        end
        
        function obj = Load(obj, dataset_name, preprocess, mode)
            %Load loading a dataset
            if (nargin < 4)
                mode = 'random'; % 'random', 'co-occur', or 'real-world'
            end
            if (nargin < 3)
                preprocess = 0; % 0 - no preprocess; 1 - normalize to [0,1] with maximum and minimum; 2 - L2-norm; 3 - zscore
            end
            
            obj.dataset_name = dataset_name;
            dir = [obj.dataset_dir, dataset_name, '/'];
            obj.dir = dir;
            dataset_file = [dir, dataset_name, '.mat'];
            load(dataset_file);
            if issparse(target)
                obj.y = full(target)';
            else
                obj.y = target'; 
            end
            if strcmp(mode, 'real-world')
                if issparse(partial_target)
                    obj.partial_y = full(partial_target)';
                else
                    obj.partial_y = partial_target';
                end
            end
            obj.X = data;
            obj.partition_file_name = [dir, obj.dataset_name];
            
            if preprocess == 1
                [n, ~]=size(obj.X);
                ma=max(obj.X);
                mi=min(obj.X);
                obj.X=(obj.X-repmat(mi,n,1))./(repmat(ma-mi,n,1)+1e-6);
                obj.X=2*obj.X./sum(obj.X, 2);
            elseif preprocess == 2
                obj.X = obj.X + eps;
                obj.X = obj.X./repmat(sqrt(sum(obj.X.^2,2)),1,size(obj.X,2));
            elseif preprocess == 3
                obj.X = zscore(obj.X);
            end
        end
        
        function [train_inds, test_inds] = Partition(obj, i, fold)
            %Partition loading a partition of dataset
            
            postfix = sprintf('_%d_train_cv%d.mat', fold, i);
            train_partition_file = [obj.partition_file_name, postfix];
            if exist(train_partition_file, 'file') == 0
                split_dataset(obj.dataset_name, fold, obj.dir)
            end
            load(train_partition_file);
            train_inds = trIdx;
            postfix = sprintf('_%d_test_cv%d.mat', fold, i);
            test_partition_file = [obj.partition_file_name, postfix];
            load(test_partition_file);
            test_inds =teIdx;
        end
        
        function [train_data, train_partial_y, test_data, test_y] = Get_set(obj, train_inds, test_inds, mode, p, r, epsilon, seed)
            if (nargin < 8)
                seed = 0;
            end
            if strcmp(mode, 'real-world')
                train_data = obj.X(train_inds, :);
                train_partial_y = obj.partial_y(train_inds, :);
                test_data = obj.X(test_inds, :);
                test_y = obj.y(test_inds, :);
            elseif strcmp(mode, 'random')
                train_data = obj.X(train_inds, :);
                train_y = obj.y(train_inds, :);
                train_partial_y = obj.random_pick(train_y, p, r, seed);
                test_data = obj.X(test_inds, :);
                test_y = obj.y(test_inds, :);
            elseif strcmp(mode, 'co-occur')
                train_data = obj.X(train_inds, :);
                train_y = obj.y(train_inds, :);
                train_partial_y = obj.co_occur(train_y, p, epsilon, seed);
                test_data = obj.X(test_inds, :);
                test_y = obj.y(test_inds, :);
            end
        end
        
        function partial_y = random_pick(obj, y, p, r, seed)
            if (nargin < 5)
                seed = 0;
            end
            rng(seed);
            [num_data, num_class] = size(y);
            partial_y = y;
            num_partial_data = round(num_data * p);
            [~, ture_label] = max(y, [], 2);
            for i = 1:num_partial_data
                other_label = setdiff([1:num_class], ture_label(i));
                temp = randperm(num_class-1);
                partial_y(i, other_label(temp(1:r))) = 1;
            end
        end
        
        function partial_y = co_occur(obj, y, p, epsilon, seed)
           if (nargin < 5)
                seed = 0;
            end
            rng(seed);
            [num_data, num_class] = size(y);
            partial_y = y;
            num_partial_data = round(num_data * p);
            [~, ture_label] = max(y, [], 2);
            dominant_label = zeros(num_class, 1);
            for j = 1:num_class
                temp = randperm(num_class);
                while(j == temp(1))
                    temp = randperm(num_class);
                end
                dominant_label(j) = temp(1);
            end
            for i = 1:num_partial_data
                other_label = setdiff([1:num_class], ture_label(i));
                rest_label = setdiff(other_label, dominant_label(ture_label(i)));
                rand_P = random('unif', 0, 1);
                if rand_P < epsilon
                    partial_y(i, dominant_label(ture_label(i))) = 1;
                else
                    temp = randperm(num_class-2);
                    partial_y(i, rest_label(temp(1))) = 1;
                end
            end
        end
    end
end

