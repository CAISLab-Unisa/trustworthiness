function [nasnetlarge_net,nasnetlarge_tl,nasnetlarge_ft] = prepare_nasnetlarge(INPUT_SIZE,OUTPUT_CLASSESS,tl_wlr,tl_blr,ft_wlr,ft_blr)
%PREPARE_nasnetlarge This function prepares three different networks.
%nasnetlarge_net will be a from-the-scratch network with weights discarded.
%nasnetlarge_tl will be a network to be used for Transfer Learning. Therefore, the learning factor of the last layer (fc6 and fc8) should be greater than other original layers (the parameters tl_wlr and tl_blr can be used to configure fc6 and fc8 learning rates).
%nasnetlarge_ft will be a network used for fine-tuning. Therefore, all the learning rates for each layer will be zero except for fc6 and fc8, which will use ft_wlr and ft_blr).
%   

% Prepare nasnetlarge for the training from scratch
nasnetlarge_net = layerGraph(nasnetlarge());
cleanNetwork(nasnetlarge_net); %clean all the weights

% Prepare nasnetlarge for Transfer Learning
nasnetlarge_tl = layerGraph(nasnetlarge());

% Prepare nasnetlarge for Fine Tuning
nasnetlarge_ft = layerGraph(nasnetlarge());

%Input layer
image_input_layer = imageInputLayer(INPUT_SIZE,Name="input_2");

%Last fully connected layer
fc_from_scratch = fullyConnectedLayer(OUTPUT_CLASSESS,Name="predictions");
fc_tl = fullyConnectedLayer(OUTPUT_CLASSESS,Name="predictions",WeightLearnRateFactor=tl_wlr,BiasLearnRateFactor=tl_blr);
fc_ft = fullyConnectedLayer(OUTPUT_CLASSESS,Name="predictions",WeightLearnRateFactor=ft_wlr,BiasLearnRateFactor=ft_blr);


%Last layer (Classificator layer)
output_layer= classificationLayer(Name="ClassificationLayer_predictions");
output_tl= classificationLayer(Name="ClassificationLayer_predictions",Classes="auto");
output_ft= classificationLayer(Name="ClassificationLayer_predictions",Classes="auto");


nasnetlarge_net=replaceLayer(nasnetlarge_net,"input_2",image_input_layer);
nasnetlarge_net=replaceLayer(nasnetlarge_net,"predictions",fc_from_scratch);   
nasnetlarge_net=replaceLayer(nasnetlarge_net,"ClassificationLayer_predictions",output_layer);

 
nasnetlarge_tl=replaceLayer(nasnetlarge_tl,"input_2",image_input_layer);
nasnetlarge_tl=replaceLayer(nasnetlarge_tl,"predictions",fc_tl);   
nasnetlarge_tl=replaceLayer(nasnetlarge_tl,"ClassificationLayer_predictions",output_tl);


nasnetlarge_ft=replaceLayer(nasnetlarge_ft,"input_2",image_input_layer);
nasnetlarge_ft=freezeNetwork(nasnetlarge_ft); %lock layers (for finetuning step)
nasnetlarge_ft=replaceLayer(nasnetlarge_ft,"predictions",fc_ft);   
nasnetlarge_ft=replaceLayer(nasnetlarge_ft,"ClassificationLayer_predictions",output_ft);
 



end

