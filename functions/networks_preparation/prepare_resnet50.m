function [resnet50_net,resnet50_tl,resnet50_ft] = prepare_resnet50(INPUT_SIZE,OUTPUT_CLASSESS,tl_wlr,tl_blr,ft_wlr,ft_blr)
%PREPARE_RESNET Summary of this function goes here
%   Detailed explanation goes here


resnet50_net = resnet50("Weights","None");
cleanNetwork(resnet50_net); %clean all the weights

resnet50_tl =  layerGraph(resnet50());
resnet50_ft =  layerGraph(resnet50());

%Input layer
image_input_layer = imageInputLayer(INPUT_SIZE,Name="input_1");
image_input_tl = imageInputLayer(INPUT_SIZE,Name="input_1");
image_input_ft = imageInputLayer(INPUT_SIZE,Name="input_1");

%Last fully connected layer
fc1000_from_scratch = fullyConnectedLayer(OUTPUT_CLASSESS,Name="fc1000");
fc1000_tl = fullyConnectedLayer(OUTPUT_CLASSESS,Name="fc1000",WeightLearnRateFactor=tl_wlr,BiasLearnRateFactor=tl_blr);
fc1000_ft = fullyConnectedLayer(OUTPUT_CLASSESS,Name="fc1000",WeightLearnRateFactor=ft_wlr,BiasLearnRateFactor=ft_blr);


%Last layer (Classificator layer)
output_layer= classificationLayer(Name="ClassificationLayer_fc1000");
output_layer_tl= classificationLayer(Name="ClassificationLayer_fc1000",Classes="auto");
output_layer_ft= classificationLayer(Name="ClassificationLayer_fc1000",Classes="auto");


resnet50_net=replaceLayer(resnet50_net,"input_1",image_input_layer);
resnet50_net=replaceLayer(resnet50_net,"fc1000",fc1000_from_scratch);
resnet50_net=replaceLayer(resnet50_net,"ClassificationLayer_fc1000",output_layer);



resnet50_tl=replaceLayer(resnet50_tl,"input_1",image_input_tl);
resnet50_tl=replaceLayer(resnet50_tl,"fc1000",fc1000_tl);
resnet50_tl=replaceLayer(resnet50_tl,"ClassificationLayer_fc1000",output_layer_tl);




resnet50_ft=replaceLayer(resnet50_ft,"input_1",image_input_ft);
resnet50_ft=freezeNetwork(resnet50_ft); %lock layers (for finetuning step)
resnet50_ft=replaceLayer(resnet50_ft,"fc1000",fc1000_ft);
resnet50_ft=replaceLayer(resnet50_ft,"ClassificationLayer_fc1000",output_layer_ft);


end

