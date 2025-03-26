function [darknet_net,darknet_tl,darknet_ft] = prepare_darknet19(INPUT_SIZE,OUTPUT_CLASSESS,tl_wlr,tl_blr,ft_wlr,ft_blr)
%PREPARE_darknet This function prepares three different networks.
%darknet_net will be a from-the-scratch network with weights discarded.
%darknet_tl will be a network to be used for Transfer Learning. Therefore, the learning factor of the last layer (fc6 and fc8) should be greater than other original layers (the parameters tl_wlr and tl_blr can be used to configure fc6 and fc8 learning rates).
%darknet_ft will be a network used for fine-tuning. Therefore, all the learning rates for each layer will be zero except for fc6 and fc8, which will use ft_wlr and ft_blr).
%   

% Prepare darknet for the training from scratch
darknet_net = darknet53("Weights","None");
cleanNetwork(darknet_net); %clean all the weights

% Prepare darknet for Transfer Learning
darknet_tl = layerGraph(darknet53());

% Prepare darknet for Fine Tuning
darknet_ft = layerGraph(darknet53());

%Input layer
image_input_layer = imageInputLayer(INPUT_SIZE,Name="input");

%Last fully connected layer
fc_from_scratch = fullyConnectedLayer(OUTPUT_CLASSESS,Name="conv53");
avg_scratch = averagePooling2dLayer([1 1],Name="avg1");

fc_layer_tl = fullyConnectedLayer(OUTPUT_CLASSESS,Name="conv53",WeightLearnRateFactor=tl_wlr,BiasLearnRateFactor=tl_blr);
avg_tl = averagePooling2dLayer([1 1],Name="avg1");

fc_layer_ft = fullyConnectedLayer(OUTPUT_CLASSESS,Name="conv53",WeightLearnRateFactor=ft_wlr,BiasLearnRateFactor=ft_blr);
avg_ft = averagePooling2dLayer([1 1],Name="avg1");



%Last layer (Classificator layer)
output_layer= classificationLayer(Name="output");
output_layer_tl= classificationLayer(Name="output",Classes="auto");
output_layer_ft= classificationLayer(Name="output",Classes="auto");


darknet_net=replaceLayer(darknet_net,"input",image_input_layer);
darknet_net=replaceLayer(darknet_net,"conv53",fc_from_scratch);   
darknet_net=replaceLayer(darknet_net,"avg1",avg_scratch);   
darknet_net=replaceLayer(darknet_net,"output",output_layer);

darknet_tl=replaceLayer(darknet_tl,"input",image_input_layer);
darknet_tl=replaceLayer(darknet_tl,"conv53",fc_layer_tl); 
darknet_tl=replaceLayer(darknet_tl,"avg1",avg_tl);   
darknet_tl=replaceLayer(darknet_tl,"output",output_layer_tl);

darknet_ft=replaceLayer(darknet_ft,"input",image_input_layer);
darknet_ft=freezeNetwork(darknet_ft); %lock layers (for finetuning step)
darknet_ft=replaceLayer(darknet_ft,"avg1",avg_ft);   
darknet_ft=replaceLayer(darknet_ft,"conv53",fc_layer_ft); %only fc6 and fc8 will learn
darknet_ft=replaceLayer(darknet_ft,"output",output_layer_ft);




end

