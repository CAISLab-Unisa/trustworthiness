function [densenet_net,densenet_tl, densenet_ft] = prepare_densenet201(INPUT_SIZE,OUTPUT_CLASSESS,tl_wlr,tl_blr,ft_wlr,ft_blr)
%PREPARE_DENSENET201 Summary of this function goes here
%   Detailed explanation goes here

% Prepare densenet201 for the training from scratch
densenet_net = densenet201("Weights","None");
cleanNetwork(densenet_net); %clean all the weights

% Prepare densenet201 for the fine tuning
densenet_tl = layerGraph(densenet201());
densenet_ft = layerGraph(densenet201());

%Input layer
image_input_layer = imageInputLayer(INPUT_SIZE,Name="input_1");

%Last fully connected layer
fc1000_from_scratch = fullyConnectedLayer(OUTPUT_CLASSESS,Name="fc1000");
fc1000_tl = fullyConnectedLayer(OUTPUT_CLASSESS,Name="fc1000",WeightLearnRateFactor=tl_wlr,BiasLearnRateFactor=tl_blr);
fc1000_ft = fullyConnectedLayer(OUTPUT_CLASSESS,Name="fc1000",WeightLearnRateFactor=ft_wlr,BiasLearnRateFactor=ft_blr);

%Last layer (Classificator layer)
output_layer_from_scratch = classificationLayer(Name="ClassificationLayer_fc1000");
output_layer_tl = classificationLayer(Name="ClassificationLayer_fc10000");
output_layer_ft = classificationLayer(Name="ClassificationLayer_fc10000");


densenet_net=replaceLayer(densenet_net,"input_1",image_input_layer);
densenet_net=replaceLayer(densenet_net,"fc1000",fc1000_from_scratch);
densenet_net=replaceLayer(densenet_net,"ClassificationLayer_fc1000",output_layer_from_scratch);

densenet_tl=replaceLayer(densenet_tl,"input_1",image_input_layer);
densenet_tl=replaceLayer(densenet_tl,"fc1000",fc1000_tl);
densenet_tl=replaceLayer(densenet_tl,"ClassificationLayer_fc1000",output_layer_tl);

densenet_ft=freezeNetwork(densenet_ft); %lock layers (for finetuning step)
densenet_ft=replaceLayer(densenet_ft,"input_1",image_input_layer);
densenet_ft=replaceLayer(densenet_ft,"fc1000",fc1000_ft);
densenet_ft=replaceLayer(densenet_ft,"ClassificationLayer_fc1000",output_layer_ft);



end

