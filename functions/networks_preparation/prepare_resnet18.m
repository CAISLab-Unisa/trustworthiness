function [resnet18_net,resnet18_tl,resnet18_ft] = prepare_resnet18(INPUT_SIZE,OUTPUT_CLASSESS,tl_wlr,tl_blr,ft_wlr,ft_blr)
%PREPARE_RESNET Summary of this function goes here
%   Detailed explanation goes here


resnet18_net = resnet18("Weights","None");
cleanNetwork(resnet18_net); %clean all the weights

resnet18_tl =  layerGraph(resnet18());
resnet18_ft =  layerGraph(resnet18());

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


resnet18_net=replaceLayer(resnet18_net,"data",image_input_layer);
resnet18_net=replaceLayer(resnet18_net,"fc1000",fc1000_from_scratch);
resnet18_net=replaceLayer(resnet18_net,"ClassificationLayer_predictions",output_layer);



resnet18_tl=replaceLayer(resnet18_tl,"data",image_input_tl);
resnet18_tl=replaceLayer(resnet18_tl,"fc1000",fc1000_tl);
resnet18_tl=replaceLayer(resnet18_tl,"ClassificationLayer_predictions",output_layer_tl);




resnet18_ft=replaceLayer(resnet18_ft,"data",image_input_ft);
resnet18_ft=freezeNetwork(resnet18_ft); %lock layers (for finetuning step)
resnet18_ft=replaceLayer(resnet18_ft,"fc1000",fc1000_ft);
resnet18_ft=replaceLayer(resnet18_ft,"ClassificationLayer_predictions",output_layer_ft);


end

