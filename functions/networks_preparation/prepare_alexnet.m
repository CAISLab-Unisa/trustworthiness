function [alexnet_net,alexnet_tl,alexnet_ft] = prepare_alexnet(INPUT_SIZE,OUTPUT_CLASSESS,tl_wlr,tl_blr,ft_wlr,ft_blr)
%PREPARE_ALEXNET This function prepares three different networks.
%alexnet_net will be a from-the-scratch network with weights discarded.
%alexnet_tl will be a network to be used for Transfer Learning. Therefore, the learning factor of the last layer (fc6 and fc8) should be greater than other original layers (the parameters tl_wlr and tl_blr can be used to configure fc6 and fc8 learning rates).
%alexnet_ft will be a network used for fine-tuning. Therefore, all the learning rates for each layer will be zero except for fc6 and fc8, which will use ft_wlr and ft_blr).
%   

% Prepare Alexnet for the training from scratch
alexnet_net = layerGraph(alexnet("Weights","None"));
cleanNetwork(alexnet_net); %clean all the weights

% Prepare Alexnet for Transfer Learning
alexnet_tl = layerGraph(alexnet());

% Prepare Alexnet for Fine Tuning
alexnet_ft = layerGraph(alexnet());

%Input layer
image_input_layer = imageInputLayer(INPUT_SIZE,Name="data");

%Last fully connected layer
fc8_from_scratch = fullyConnectedLayer(OUTPUT_CLASSESS,Name="fc8");
fc8_layer_tl = fullyConnectedLayer(OUTPUT_CLASSESS,Name="fc8",WeightLearnRateFactor=tl_wlr,BiasLearnRateFactor=tl_blr);
fc6_layer_tl = fullyConnectedLayer(4096,Name="fc6",WeightLearnRateFactor=tl_wlr,BiasLearnRateFactor=tl_blr);

fc8_layer_ft = fullyConnectedLayer(OUTPUT_CLASSESS,Name="fc8",WeightLearnRateFactor=ft_wlr,BiasLearnRateFactor=ft_blr);
fc6_layer_ft = fullyConnectedLayer(4096,Name="fc6",WeightLearnRateFactor=ft_wlr,BiasLearnRateFactor=ft_blr);


%Last layer (Classificator layer)
output_layer= classificationLayer(Name="output");
output_layer_tl= classificationLayer(Name="output",Classes="auto");
output_layer_ft= classificationLayer(Name="output",Classes="auto");


alexnet_net=replaceLayer(alexnet_net,"data",image_input_layer);
alexnet_net=replaceLayer(alexnet_net,"fc8",fc8_from_scratch);   
alexnet_net=replaceLayer(alexnet_net,"output",output_layer);

alexnet_tl=replaceLayer(alexnet_tl,"data",image_input_layer);
alexnet_tl=replaceLayer(alexnet_tl,"fc6",fc6_layer_tl); 
alexnet_tl=replaceLayer(alexnet_tl,"fc8",fc8_layer_tl);
alexnet_tl=replaceLayer(alexnet_tl,"output",output_layer_tl);

alexnet_ft=freezeNetwork(alexnet_ft); %lock layers (for finetuning step)

alexnet_ft=replaceLayer(alexnet_ft,"data",image_input_layer);
alexnet_ft=replaceLayer(alexnet_ft,"fc6",fc6_layer_ft); %only fc6 and fc8 will learn
alexnet_ft=replaceLayer(alexnet_ft,"fc8",fc8_layer_ft);
alexnet_ft=replaceLayer(alexnet_ft,"output",output_layer_ft);




end

