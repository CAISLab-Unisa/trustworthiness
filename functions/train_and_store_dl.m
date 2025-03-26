function [trained_nn,trained_nn_score] = train_and_store_dl(dataset_name,ds,ds4perf,ds_split_ratio,nn_name,nn,train_options)
%TRAIN_AND_STORE Summary of this function goes here
%   Detailed explanation goes here

    ds_train = ds{1};
    
    
    % Contains the split configuration for the train, validation and test sets
    ds_split_prefix = strcat(string(ds_split_ratio(1)),"_",string(ds_split_ratio(2)),"_",string(ds_split_ratio(3)));

    % Prepares path where trained networks will be saved
    store_path = strcat("./Trained/",dataset_name);    
    store_create_path(store_path);
    store_path = strcat("./Trained/",dataset_name,"/",nn_name);    
    store_create_path(store_path);
    store_path = strcat("./Trained/",dataset_name,"/",nn_name,"/",ds_split_prefix);
    store_create_path(store_path);

    %Execute the Training Process
    gpuDevice(1);
    [trained_nn,trained_nn_score] = trainnet(ds_train,nn,"crossentropy",train_options);  
  
    %Save trained network and all the training data and the training plot
    save(strcat(store_path,"/trained_network.mat"),"trained_nn");
    save(strcat(store_path,"/training_score.mat"),"trained_nn_score")
    %save_training_plot(strcat(store_path,"/TrainingPlot.png"))
  

 
    store_performance_dl(dataset_name,nn_name,trained_nn,ds_split_ratio,ds4perf);
    store_gradcam(dataset_name,nn_name,trained_nn,"test",ds_split_ratio,ds4perf{3},"predict","gpu");
    
  % store_gradcam(dataset_name,nn_name,trained_nn,"train",ds_split_ratio,ds4perf{1},"classify","gpu");
  % store_gradcam(dataset_name,nn_name,trained_nn,"validation",ds_split_ratio,ds4perf{2},"classify","gpu");
   



end

