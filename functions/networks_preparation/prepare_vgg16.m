function [vgg16_net,vgg16_tl,vgg16_ft] = prepare_vgg16(INPUT_SIZE,OUTPUT_CLASSESS,tl_wlr,tl_blr,ft_wlr,ft_blr)
%PREPARE_vgg16 This function prepares three different networks.
%vgg16_net will be a from-the-scratch network with weights discarded.
%vgg16_tl will be a network to be used for Transfer Learning. Therefore, the learning factor of the last layer (fc6 and fc8) should be greater than other original layers (the parameters tl_wlr and tl_blr can be used to configure fc6 and fc8 learning rates).
%vgg16_ft will be a network used for fine-tuning. Therefore, all the learning rates for each layer will be zero except for fc6 and fc8, which will use ft_wlr and ft_blr).
%   

% Prepare vgg16 for the training from scratch
vgg16_net = layerGraph(vgg16());
cleanNetwork(vgg16_net); %clean all the weights

% Prepare vgg16 for Transfer Learning
vgg16_tl = layerGraph(vgg16());

% Prepare vgg16 for Fine Tuning
vgg16_ft = layerGraph(vgg16());

%Input layer
image_input_layer = imageInputLayer(INPUT_SIZE,Name="input");

%Last fully connected layer
fc_from_scratch = fullyConnectedLayer(OUTPUT_CLASSESS,Name="fc8");
fc_tl = fullyConnectedLayer(OUTPUT_CLASSESS,Name="fc8",WeightLearnRateFactor=tl_wlr,BiasLearnRateFactor=tl_blr);
fc_ft = fullyConnectedLayer(OUTPUT_CLASSESS,Name="fc8",WeightLearnRateFactor=ft_wlr,BiasLearnRateFactor=ft_blr);


%Last layer (Classificator layer)
output_layer= classificationLayer(Name="output");
output_tl= classificationLayer(Name="output",Classes="auto");
output_ft= classificationLayer(Name="output",Classes="auto");


vgg16_net=replaceLayer(vgg16_net,"input",image_input_layer);
vgg16_net=replaceLayer(vgg16_net,"fc8",fc_from_scratch);   
vgg16_net=replaceLayer(vgg16_net,"output",output_layer);

 
vgg16_tl=replaceLayer(vgg16_tl,"input",image_input_layer);
vgg16_tl=replaceLayer(vgg16_tl,"fc8",fc_tl);   
vgg16_tl=replaceLayer(vgg16_tl,"output",output_tl);


vgg16_ft=replaceLayer(vgg16_ft,"input",image_input_layer);
vgg16_ft=freezeNetwork(vgg16_ft); %lock layers (for finetuning step)
vgg16_ft=replaceLayer(vgg16_ft,"fc8",fc_ft);   
vgg16_ft=replaceLayer(vgg16_ft,"output",output_ft);
 



end

