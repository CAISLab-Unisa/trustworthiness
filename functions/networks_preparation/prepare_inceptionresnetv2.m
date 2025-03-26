ction [inceptionresnetv2_net,inceptionresnetv2_tl,inceptionresnetv2_ft] = prepare_inceptionresnetv2(INPUT_SIZE,OUTPUT_CLASSESS,tl_wlr,tl_blr,ft_wlr,ft_blr)
%PREPARE_inceptionresnetv2 Summary of this function goes here
%   Detailed explanation goes here

% Prepare inceptionresnetv2 for the training from scratch
inceptionresnetv2_net = inceptionresnetv2("Weights","None");
cleanNetwork(inceptionresnetv2_net);
% Prepare inceptionresnetv2 for the fine tuning
inceptionresnetv2_tl= layerGraph(inceptionresnetv2());
inceptionresnetv2_ft= layerGraph(inceptionresnetv2());


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


inceptionresnetv2_net=replaceLayer(inceptionresnetv2_net,"input_1",image_input_layer);
inceptionresnetv2_net=replaceLayer(inceptionresnetv2_net,"predictions",fc8_from_scratch);
inceptionresnetv2_net=replaceLayer(inceptionresnetv2_net,"ClassificationLayer_predictions",output_layer);

inceptionresnetv2_tl=replaceLayer(inceptionresnetv2_tl,"input_1",image_input_layer);
inceptionresnetv2_tl=replaceLayer(inceptionresnetv2_tl,"predictions",fc8_layer_tl);
inceptionresnetv2_tl=replaceLayer(inceptionresnetv2_tl,"ClassificationLayer_predictions",output_layer_tl);


inceptionresnetv2_ft=replaceLayer(inceptionresnetv2_ft,"input_1",image_input_layer);
freezeNetwork(inceptionresnetv2_ft);
inceptionresnetv2_ft=replaceLayer(inceptionresnetv2_ft,"predictions",fc8_layer_ft);
inceptionresnetv2_ft=replaceLayer(inceptionresnetv2_ft,"ClassificationLayer_predictions",output_layer_ft);



end

