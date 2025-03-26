function [] = store_gradcam(dataset_name,net_name,cnn_network,dataset_prefix,ds_split_ratio,test,fcnToUse,execEnv)
%STORE_GRADCAM Summary of this function goes here
%   This function extract GRADCAM using the trained 'cnn_network' on the
%   'test' dataImageStore.




% Contains the split configuration for the train, validation and test sets
ds_split_prefix = strcat(string(ds_split_ratio(1)),"_",string(ds_split_ratio(2)),"_",string(ds_split_ratio(3)));

%RESULTS / DATASET / NET_NAME
store_path = strcat("./results/",dataset_name,"/",net_name);    
store_create_path(store_path);

%RESULTS / DATASET / NET_NAME / (DATASET SPLIT CONFIG)
store_path = strcat("./results/",dataset_name,"/",net_name,"/",ds_split_prefix);
store_create_path(store_path);

%RESULTS / DATASET / NET_NAME / (DATASET SPLIT CONFIG) / DS_PREFIX
store_path = strcat("./results/",dataset_name,"/",net_name,"/",ds_split_prefix,"/",dataset_prefix);
store_create_path(store_path);

%RESULTS / DATASET / NET_NAME / (DATASET SPLIT CONFIG) / DS_PREFIX /
%gradcam
store_path = strcat("./results/",dataset_name,"/",net_name,"/",ds_split_prefix,"/",dataset_prefix,"/gradcam");
store_create_path(store_path);

%Directory to create (Iterate by Labels)
lab = unique(test.Labels);
for i = 1:size(unique(lab),1)
    store_path = strcat("./results/",dataset_name,"/",net_name,"/",ds_split_prefix,"/",dataset_prefix,"/gradcam/",char(lab(i)));
    store_create_path(store_path);
end




   

for file_index = 1:length(test.Files)
    
    image_name = test.Files(file_index);
    image_label = test.Labels(file_index);
    [a,image_name,image_ext] = fileparts(image_name);
    
    store_path = strcat("./results/",dataset_name,"/",net_name,"/",ds_split_prefix,"/",dataset_prefix,"/gradcam/",string(image_label),"/",image_name,image_ext);
    if exist(store_path, 'file')
        continue
    end

    X = readimage(test,file_index);
    if fcnToUse=="classify"
    label=classify(cnn_network,X);
    else
    label=minibatchpredict(cnn_network,X);
    end 


    scoreMap = gradCAM(cnn_network,X,label,ExecutionEnvironment=execEnv);
     

    figure("Visible","off");
    imshow(X);
    hold on;
    imagesc(scoreMap,'AlphaData',0.3);
    colormap jet;
    f = gcf;
    exportgraphics(f,store_path);
    hold off;
    
    
    
end

end

