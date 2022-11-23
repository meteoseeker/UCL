function split_dataset(dataset_name, fold, dataset_dir)
    load([dataset_dir, dataset_name, '.mat']);
    
    N = size(data, 1);
    CVO = cvpartition(N,'KFold',10);
    partition_file_name = [dataset_dir, dataset_name];
    for i = 1:CVO.NumTestSets
        trIdx = CVO.training(i);
        teIdx = CVO.test(i);
        
        postfix = sprintf('_%d_train_cv%d.mat', fold, i);
        train_partition_file = [partition_file_name, postfix];
        save(train_partition_file, 'trIdx');
        postfix = sprintf('_%d_test_cv%d.mat', fold, i);
        test_partition_file = [partition_file_name, postfix];
        save(test_partition_file, 'teIdx');
    end
end