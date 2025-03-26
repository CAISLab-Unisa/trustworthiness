function [xception_net,xception_tl,xception_ft] = prepare_xception(INPUT_SIZE,OUTPUT_CLASSESS,tl_wlr,tl_blr,ft_wlr,ft_blr)
%PREPARE_xception Summary of this function goes here
%   Detailed explanation goes here

% Prepare xception for the training from scratch
xception_net = xception("Weights","None");
cleanNetwork(xception_net);
% Prepare xception for the fine tuning
xception_tl= layerGraph(xception());
xception_ft= layerGraph(xception());


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


xception_net=replaceLayer(xception_net,"input_1",image_input_layer);
xception_net=replaceLayer(xception_net,"predictions",fc8_from_scratch);
xception_net=replaceLayer(xception_net,"ClassificationLayer_predictions",output_layer);

xception_tl=replaceLayer(xception_tl,"input_1",image_input_layer);
xception_tl=replaceLayer(xception_tl,"predictions",fc8_layer_tl);
xception_tl=replaceLayer(xception_tl,"ClassificationLayer_predictions",output_layer_tl);


xception_ft=replaceLayer(xception_ft,"input_1",image_input_layer);
freezeNetwork(xception_ft);
xception_ft=replaceLayer(xception_ft,"predictions",fc8_layer_ft);
xception_ft=replaceLayer(xception_ft,"ClassificationLayer_predictions",output_layer_ft);



end

