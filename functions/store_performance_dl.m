function [] = store_performance_dl(dataset_name,net_name,cnn_network,ds_split_ratio,datasets)
%store_performance 



ds_split_prefix = strcat(string(ds_split_ratio(1)),"_",string(ds_split_ratio(2)),"_",string(ds_split_ratio(3)));


store_path = strcat("./results/",dataset_name,"/");
store_create_path(store_path);
store_path = strcat("./results/",dataset_name,"/Performance.tsv");
 
if ~exist(store_path, 'file')
      fileID = fopen(store_path,"w");
      fprintf(fileID,"Dataset Name \t");
        fprintf(fileID,"Train ratio \t");
        fprintf(fileID,"Validation ratio \t");
        fprintf(fileID,"Test ratio \t");
        fprintf(fileID,"NN Name \t");
        fprintf(fileID,"Accuracy (Training)\t");
        fprintf(fileID,"Accuracy (Validation)\t");
        fprintf(fileID,"Accuracy (Test)\n");
else
      fileID = fopen(store_path,"a");
end


fprintf(fileID,"%s\t",dataset_name);
fprintf(fileID,"%f\t",ds_split_ratio(1));
fprintf(fileID,"%f\t",ds_split_ratio(2));
fprintf(fileID,"%f\t",ds_split_ratio(3));
fprintf(fileID,"%s\t",net_name);

store_create_path(strcat("./results/",dataset_name,"/",net_name,"/",ds_split_prefix));

%Access to splitted datasets
train = datasets{1};
validation = datasets{2};
test = datasets{3};


%%% Esegue la validazione per l-oggeto dl_network
classNames = categories(train.Labels);
mbq = minibatchqueue(train,1, ...
    MiniBatchFormat="SSCB");
YTest = [];
% Loop over mini-batches.
while hasdata(mbq)
    
    % Read mini-batch of data.
    X = next(mbq);
       
    % Make predictions using the predict function.
    Y = predict(cnn_network,X);
   
    % Convert scores to classes.
    predBatch = onehotdecode(Y,classNames,1);
    YTest = [YTest; predBatch'];
end
TTest = train.Labels;
VALccuracy = mean(YTest == TTest)
fprintf(fileID,"%f\t",VALccuracy);

CM = confusionmat(TTest,YTest);
dlmwrite(strcat("./results/",dataset_name,"/",net_name,"/",ds_split_prefix,"/cm_training.txt"),CM);

figure
confusionchart(TTest,YTest,'ColumnSummary','column-normalized',...
              'RowSummary','row-normalized','Title','Confusion Chart for TRAINING SET');

store_path = strcat("./results/",dataset_name,"/",net_name,"/",ds_split_prefix,"/cm_training.png");
f=gcf;
exportgraphics(f,store_path);


%% controlla il validation set

classNames = categories(validation.Labels);
mbq = minibatchqueue(validation,1, ...
    MiniBatchFormat="SSCB");
YTest = [];
% Loop over mini-batches.
while hasdata(mbq)
    
    % Read mini-batch of data.
    X = next(mbq);
       
    % Make predictions using the predict function.
    Y = predict(cnn_network,X);
   
    % Convert scores to classes.
    predBatch = onehotdecode(Y,classNames,1);
    YTest = [YTest; predBatch'];
end
TTest = validation.Labels;
VALccuracy = mean(YTest == TTest)
fprintf(fileID,"%f\t",VALccuracy);

CM = confusionmat(TTest,YTest);
dlmwrite(strcat("./results/",dataset_name,"/",net_name,"/",ds_split_prefix,"/cm_validation.txt"),CM);

figure
confusionchart(TTest,YTest,'ColumnSummary','column-normalized',...
              'RowSummary','row-normalized','Title','Confusion Chart for VALIDATION SET');

store_path = strcat("./results/",dataset_name,"/",net_name,"/",ds_split_prefix,"/cm_validation.png");
f=gcf;
exportgraphics(f,store_path);



%% valida il test set
classNames = categories(test.Labels);
mbq = minibatchqueue(test,1, ...
    MiniBatchFormat="SSCB");
YTest = [];
% Loop over mini-batches.
while hasdata(mbq)
    
    % Read mini-batch of data.
    X = next(mbq);
       
    % Make predictions using the predict function.
    Y = predict(cnn_network,X);
   
    % Convert scores to classes.
    predBatch = onehotdecode(Y,classNames,1);
    YTest = [YTest; predBatch'];
end
TTest = test.Labels;
VALccuracy = mean(YTest == TTest)
fprintf(fileID,"%f\t",VALccuracy);

CM = confusionmat(TTest,YTest);
dlmwrite(strcat("./results/",dataset_name,"/",net_name,"/",ds_split_prefix,"/cm_test.txt"),CM);

figure
confusionchart(TTest,YTest,'ColumnSummary','column-normalized',...
              'RowSummary','row-normalized','Title','Confusion Chart for TEST SET');
store_path = strcat("./results/",dataset_name,"/",net_name,"/",ds_split_prefix,"/cm_test.png");
f=gcf;
exportgraphics(f,store_path);



fclose(fileID);

end

