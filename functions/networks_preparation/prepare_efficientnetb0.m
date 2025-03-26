function [efficientnetb0_net,efficientnetb0_tl,efficientnetb0_ft] = prepare_efficientnetb0(INPUT_SIZE,OUTPUT_CLASSESS,tl_wlr,tl_blr,ft_wlr,ft_blr)
%PREPARE_efficientnetb0 This function prepares three different networks.
%efficientnetb0_net will be a from-the-scratch network with weights discarded.
%efficientnetb0_tl will be a network to be used for Transfer Learning. Therefore, the learning factor of the last layer (fc6 and fc8) should be greater than other original layers (the parameters tl_wlr and tl_blr can be used to configure fc6 and fc8 learning rates).
%efficientnetb0_ft will be a network used for fine-tuning. Therefore, all the learning rates for each layer will be zero except for fc6 and fc8, which will use ft_wlr and ft_blr).
%   

% Prepare efficientnetb0 for the training from scratch
efficientnetb0_net = layerGraph(efficientnetb0());
cleanNetwork(efficientnetb0_net); %clean all the weights

% Prepare efficientnetb0 for Transfer Learning
efficientnetb0_tl = layerGraph(efficientnetb0());

% Prepare efficientnetb0 for Fine Tuning
efficientnetb0_ft = layerGraph(efficientnetb0());

%Input layer
image_input_layer = imageInputLayer(INPUT_SIZE,Name="ImageInput");

%Last fully connected layer
fc_from_scratch = fullyConnectedLayer(OUTPUT_CLASSESS,Name="efficientnet-b0|model|head|dense|MatMul");
fc_tl = fullyConnectedLayer(OUTPUT_CLASSESS,Name="efficientnet-b0|model|head|dense|MatMul",WeightLearnRateFactor=tl_wlr,BiasLearnRateFactor=tl_blr);
fc_ft = fullyConnectedLayer(OUTPUT_CLASSESS,Name="efficientnet-b0|model|head|dense|MatMul",WeightLearnRateFactor=ft_wlr,BiasLearnRateFactor=ft_blr);


%Last layer (Classificator layer)
output_layer= classificationLayer(Name="classification");
output_tl= classificationLayer(Name="classification",Classes="auto");
output_ft= classificationLayer(Name="classification",Classes="auto");


efficientnetb0_net=replaceLayer(efficientnetb0_net,"ImageInput",image_input_layer);
efficientnetb0_net=replaceLayer(efficientnetb0_net,"efficientnet-b0|model|head|dense|MatMul",fc_from_scratch);   
efficientnetb0_net=replaceLayer(efficientnetb0_net,"classification",output_layer);

 
efficientnetb0_tl=replaceLayer(efficientnetb0_tl,"ImageInput",image_input_layer);
efficientnetb0_tl=replaceLayer(efficientnetb0_tl,"efficientnet-b0|model|head|dense|MatMul",fc_tl);   
efficientnetb0_tl=replaceLayer(efficientnetb0_tl,"classification",output_tl);


efficientnetb0_ft=replaceLayer(efficientnetb0_ft,"ImageInput",image_input_layer);
efficientnetb0_ft=freezeNetwork(efficientnetb0_ft); %lock layers (for finetuning step)
efficientnetb0_ft=replaceLayer(efficientnetb0_ft,"efficientnet-b0|model|head|dense|MatMul",fc_ft);   
efficientnetb0_ft=replaceLayer(efficientnetb0_ft,"classification",output_ft);
 



end

