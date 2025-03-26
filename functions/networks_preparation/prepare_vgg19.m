function [vgg19_net,vgg19_tl,vgg19_ft] = prepare_vgg19(INPUT_SIZE,OUTPUT_CLASSESS,tl_wlr,tl_blr,ft_wlr,ft_blr)
%PREPARE_vgg19 This function prepares three different networks.
%vgg19_net will be a from-the-scratch network with weights discarded.
%vgg19_tl will be a network to be used for Transfer Learning. Therefore, the learning factor of the last layer (fc6 and fc8) should be greater than other original layers (the parameters tl_wlr and tl_blr can be used to configure fc6 and fc8 learning rates).
%vgg19_ft will be a network used for fine-tuning. Therefore, all the learning rates for each layer will be zero except for fc6 and fc8, which will use ft_wlr and ft_blr).
%   

% Prepare vgg19 for the training from scratch
vgg19_net = layerGraph(vgg19());
cleanNetwork(vgg19_net); %clean all the weights

% Prepare vgg19 for Transfer Learning
vgg19_tl = layerGraph(vgg19());

% Prepare vgg19 for Fine Tuning
vgg19_ft = layerGraph(vgg19());

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


vgg19_net=replaceLayer(vgg19_net,"input",image_input_layer);
vgg19_net=replaceLayer(vgg19_net,"fc8",fc_from_scratch);   
vgg19_net=replaceLayer(vgg19_net,"output",output_layer);

 
vgg19_tl=replaceLayer(vgg19_tl,"input",image_input_layer);
vgg19_tl=replaceLayer(vgg19_tl,"fc8",fc_tl);   
vgg19_tl=replaceLayer(vgg19_tl,"output",output_tl);


vgg19_ft=replaceLayer(vgg19_ft,"input",image_input_layer);
vgg19_ft=freezeNetwork(vgg19_ft); %lock layers (for finetuning step)
vgg19_ft=replaceLayer(vgg19_ft,"fc8",fc_ft);   
vgg19_ft=replaceLayer(vgg19_ft,"output",output_ft);
 



end

