function [nasnetmobile_net,nasnetmobile_tl,nasnetmobile_ft] = prepare_nasnetmobile(INPUT_SIZE,OUTPUT_CLASSESS,tl_wlr,tl_blr,ft_wlr,ft_blr)
%PREPARE_NASNETMOBILE This function prepares three different networks.
%nasnetmobile_net will be a from-the-scratch network with weights discarded.
%nasnetmobile_tl will be a network to be used for Transfer Learning. Therefore, the learning factor of the last layer (fc6 and fc8) should be greater than other original layers (the parameters tl_wlr and tl_blr can be used to configure fc6 and fc8 learning rates).
%nasnetmobile_ft will be a network used for fine-tuning. Therefore, all the learning rates for each layer will be zero except for fc6 and fc8, which will use ft_wlr and ft_blr).
%   

% Prepare nasnetmobile for the training from scratch
nasnetmobile_net = layerGraph(nasnetmobile());
cleanNetwork(nasnetmobile_net); %clean all the weights

% Prepare nasnetmobile for Transfer Learning
nasnetmobile_tl = layerGraph(nasnetmobile());

% Prepare nasnetmobile for Fine Tuning
nasnetmobile_ft = layerGraph(nasnetmobile());

%Input layer
image_input_layer = imageInputLayer(INPUT_SIZE,Name="input_1");

%Last fully connected layer
fc_from_scratch = fullyConnectedLayer(OUTPUT_CLASSESS,Name="predictions");
fc_tl = fullyConnectedLayer(OUTPUT_CLASSESS,Name="predictions",WeightLearnRateFactor=tl_wlr,BiasLearnRateFactor=tl_blr);
fc_ft = fullyConnectedLayer(OUTPUT_CLASSESS,Name="predictions",WeightLearnRateFactor=ft_wlr,BiasLearnRateFactor=ft_blr);


%Last layer (Classificator layer)
output_layer= classificationLayer(Name="ClassificationLayer_predictions");
output_tl= classificationLayer(Name="ClassificationLayer_predictions",Classes="auto");
output_ft= classificationLayer(Name="ClassificationLayer_predictions",Classes="auto");


nasnetmobile_net=replaceLayer(nasnetmobile_net,"input_1",image_input_layer);
nasnetmobile_net=replaceLayer(nasnetmobile_net,"predictions",fc_from_scratch);   
nasnetmobile_net=replaceLayer(nasnetmobile_net,"ClassificationLayer_predictions",output_layer);

 
nasnetmobile_tl=replaceLayer(nasnetmobile_tl,"input_1",image_input_layer);
nasnetmobile_tl=replaceLayer(nasnetmobile_tl,"predictions",fc_tl);   
nasnetmobile_tl=replaceLayer(nasnetmobile_tl,"ClassificationLayer_predictions",output_tl);


nasnetmobile_ft=replaceLayer(nasnetmobile_ft,"input_1",image_input_layer);
nasnetmobile_ft=freezeNetwork(nasnetmobile_ft); %lock layers (for finetuning step)
nasnetmobile_ft=replaceLayer(nasnetmobile_ft,"predictions",fc_ft);   
nasnetmobile_ft=replaceLayer(nasnetmobile_ft,"ClassificationLayer_predictions",output_ft);
 



end

