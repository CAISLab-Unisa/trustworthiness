function [googlenet_net,googlenet_tl,googlenet_ft] = prepare_googlenet(INPUT_SIZE,OUTPUT_CLASSESS,tl_wlr,tl_blr,ft_wlr,ft_blr)
%PREPARE_googlenet Summary of this function goes here
%   Detailed explanation goes here

% Prepare googlenet for the training from scratch
googlenet_net = googlenet("Weights","None");
cleanNetwork(googlenet_net); %clean all the weights
% Prepare googlenet for the fine tuning
googlenet_tl = layerGraph(googlenet());
googlenet_ft = layerGraph(googlenet());

%Input layer
image_input_layer = imageInputLayer(INPUT_SIZE,Name="data");

%Last fully connected layer
fc8_from_scratch = fullyConnectedLayer(OUTPUT_CLASSESS,Name="loss3-classifier");
fc8_layer_tl = fullyConnectedLayer(OUTPUT_CLASSESS,Name="loss3-classifier",WeightLearnRateFactor=tl_wlr,BiasLearnRateFactor=tl_blr);
fc8_layer_ft = fullyConnectedLayer(OUTPUT_CLASSESS,Name="loss3-classifier",WeightLearnRateFactor=ft_wlr,BiasLearnRateFactor=ft_blr);

%Last layer (Classificator layer)
output_layer= classificationLayer(Name="output");
output_layer_tl= classificationLayer(Name="output",Classes="auto");
output_layer_ft= classificationLayer(Name="output",Classes="auto");


googlenet_net=replaceLayer(googlenet_net,"data",image_input_layer);
googlenet_net=replaceLayer(googlenet_net,"loss3-classifier",fc8_from_scratch);
googlenet_net=replaceLayer(googlenet_net,"output",output_layer);

googlenet_tl=replaceLayer(googlenet_tl,"data",image_input_layer);
googlenet_tl=replaceLayer(googlenet_tl,"loss3-classifier",fc8_layer_tl);
googlenet_tl=replaceLayer(googlenet_tl,"output",output_layer_tl);

googlenet_ft=freezeNetwork(googlenet_ft); %lock layers (for finetuning step)
googlenet_ft=replaceLayer(googlenet_ft,"data",image_input_layer);
googlenet_ft=replaceLayer(googlenet_ft,"loss3-classifier",fc8_layer_ft);
googlenet_ft=replaceLayer(googlenet_ft,"output",output_layer_ft);


end

