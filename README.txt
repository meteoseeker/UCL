We give three sample files for our proposed approach as well as two extensional approaches:
'Exp_SURE_LIFT_Spectral_3P.m' for UCL, 
'Exp_KNN_LIFT_Spectral_3P.m' for UCL-KNN, 
'Exp_SVM_LIFT_Spectral_3P.m' for UCL-SVM.
You can direcly run each of them.

We present two datasests (the real-world dataset 'lost' and the UCI dataset 'zoo') in 'Datasets/'. 
You can add more datasets in this diarectory.
Noted that UCI datasets are used for experiments on synthetic datasets.


------------to varify UCL:----------------

Take the real-world dataset 'lost' for example:
1) choose file 'Exp_SURE_LIFT_Spectral_3P.m' 
2) set dataset_names = {'lost'};
3) set mode = 'real-world' and run.
After the process is complete, you can get the following results:
a) In directory 'Datasets/lost/', ten splited datasets are recorded for ten-fold validation. 
b) there will be a new generated dirctory 'Exp_UCL', where you can find a log recorded with accuracy and running time in 'Exp_UCL/Log' and 
find generated results in 'Exp_UCL/res'


Take the synthetic dataset 'glass' for example:
1) choose file 'Exp_SURE_LIFT_Spectral_3P.m'
2) set dataset_names = {'glass'};

    There are two configurations: 
    for configuration I in our paper:                 
    1) set mode = 'random' 
    2) set num_FL (number of false postive labels) as [1, 2, 3] and run
    for configuration II in our paper:   
    1) set mode = 'co-occur'
    2) set epsilon (probability for picking co-occurring label as false postive label) as [0.1, 0.2, ..., 0.7] and run

After the process is complete, you can get the following results:
a) In directory 'Datasets/lost', ten splited datasets are recorded for ten-fold validation. 
b) there will be a new generated directory 'Exp_UCL', where you can find a log recorded with accuracy and running time in 'Exp_UCL/Log' and 
find generated results in 'Exp_UCL/res'


-------------to verify UCL-KNN:-------------
Altering sample file to 'Exp_KNN_LIFT_Spectral_3P.m', the rest opration is same as varyfying UCL


-------------to verify UCL-KNN:-------------
Altering sample file to 'Exp_SVM_LIFT_Spectral_3P.m', the rest opration is same as varyfying UCL 


-------------main processes-----------------
1)split data for experiments
correspongding files are in 'tools/'
1)function 'PL_LIFT_Spectral_3P' conduct graph-based label enhancement and generate unaware label specific features.
correspongding files are in 'algorithms/PL_LIFT/'
2)new formed data is trained and predicted by PLL classifiers: SURE(default), KNN, SVM.
correspongding files are in 'algorithms/PL_classifiers_LS/'

-------------other notification------------
Pacage libsvm-3.24 is needed, we include the pacage in main directory for you 



