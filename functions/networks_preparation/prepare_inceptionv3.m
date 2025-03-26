function [inceptionv3_net,inceptionv3_tl,inceptionv3_ft] = prepare_inceptionv3(INPUT_SIZE,OUTPUT_CLASSESS,tl_wlr,tl_blr,ft_wlr,ft_blr)
%PREPARE_inceptionv3 Summary of this function goes here
%   Detailed explanation goes here

% Prepare inceptionv3 for the training from scratch
inceptionv3_net = inceptionv3("Weights","None");
cleanNetwork(inceptionv3_net);
% Prepare inceptionv3 for the fine tuning
inceptionv3_tl= layerGraph(inceptionv3());
inceptionv3_ft= layerGraph(inceptionv3());


%Input layer
image_input_layer = imageInputLayer(INPUT_SIZE,Name="input_1");

%Last fully connected layer
fc8_from_scratch = fullyConnectedLayer(OUTPUT_CLASSESS,Name="predictions");
fc8_layer_tl = fullyConnectedLayer(OUTPUT_CLASSESS,Name="predictions",WeightLearnRateFactor=tl_wlr,BiasLearnRateFactor=tl_blr);
fc8_layer_ft = fullyConnectedLayer(OUTPUT_CLASSESS,Name="predictions",WeightLearnRateFactor=ft_wlr,BiasLearnRateFactor=ft_blr);


%Last layer (Classificator layer)
output_layer= classificationLayer(Name="ClassificationLayer_predictions");
output_layer_tl= classificationLayer(Name="ClassificationLayer_predictions",Classes="auto");
output_layer_ft= classificationLayer(Name="ClassificationLayer_predictions",Classes="auto");


inceptionv3_net=replaceLayer(inceptionv3_net,"input_1",image_input_layer);
inceptionv3_net=replaceLayer(inceptionv3_net,"predictions",fc8_from_scratch);
inceptionv3_net=replaceLayer(inceptionv3_net,"ClassificationLayer_predictions",output_layer);

inceptionv3_tl=replaceLayer(inceptionv3_tl,"input_1",image_input_layer);
inceptionv3_tl=replaceLayer(inceptionv3_tl,"predictions",fc8_layer_tl);
inceptionv3_tl=replaceLayer(inceptionv3_tl,"ClassificationLayer_predictions",output_layer_tl);


inceptionv3_ft=replaceLayer(inceptionv3_ft,"input_1",image_input_layer);
freezeNetwork(inceptionv3_ft);
inceptionv3_ft=replaceLayer(inceptionv3_ft,"predictions",fc8_layer_ft);
inceptionv3_ft=replaceLayer(inceptionv3_ft,"ClassificationLayer_predictions",output_layer_ft);



end

