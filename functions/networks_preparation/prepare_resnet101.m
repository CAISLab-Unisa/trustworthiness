function [resnet101_net,resnet101_tl,resnet101_ft] = prepare_resnet101(INPUT_SIZE,OUTPUT_CLASSESS,tl_wlr,tl_blr,ft_wlr,ft_blr)
%PREPARE_RESNET Summary of this function goes here
%   Detailed explanation goes here


resnet101_net = resnet101("Weights","None");
cleanNetwork(resnet101_net); %clean all the weights

resnet101_tl =  layerGraph(resnet101());
resnet101_ft =  layerGraph(resnet101());

%Input layer
image_input_layer = imageInputLayer(INPUT_SIZE,Name="data");
image_input_tl = imageInputLayer(INPUT_SIZE,Name="data");
image_input_ft = imageInputLayer(INPUT_SIZE,Name="data");

%Last fully connected layer
fc1000_from_scratch = fullyConnectedLayer(OUTPUT_CLASSESS,Name="fc1000");
fc1000_tl = fullyConnectedLayer(OUTPUT_CLASSESS,Name="fc1000",WeightLearnRateFactor=tl_wlr,BiasLearnRateFactor=tl_blr);
fc1000_ft = fullyConnectedLayer(OUTPUT_CLASSESS,Name="fc1000",WeightLearnRateFactor=ft_wlr,BiasLearnRateFactor=ft_blr);


%Last layer (Classificator layer)
output_layer= classificationLayer(Name="ClassificationLayer_predictions");
output_layer_tl= classificationLayer(Name="ClassificationLayer_predictions",Classes="auto");
output_layer_ft= classificationLayer(Name="ClassificationLayer_predictions",Classes="auto");


resnet101_net=replaceLayer(resnet101_net,"data",image_input_layer);
resnet101_net=replaceLayer(resnet101_net,"fc1000",fc1000_from_scratch);
resnet101_net=replaceLayer(resnet101_net,"ClassificationLayer_predictions",output_layer);



resnet101_tl=replaceLayer(resnet101_tl,"data",image_input_tl);
resnet101_tl=replaceLayer(resnet101_tl,"fc1000",fc1000_tl);
resnet101_tl=replaceLayer(resnet101_tl,"ClassificationLayer_predictions",output_layer_tl);




resnet101_ft=replaceLayer(resnet101_ft,"data",image_input_ft);
resnet101_ft=freezeNetwork(resnet101_ft); %lock layers (for finetuning step)
resnet101_ft=replaceLayer(resnet101_ft,"fc1000",fc1000_ft);
resnet101_ft=replaceLayer(resnet101_ft,"ClassificationLayer_predictions",output_layer_ft);


end

