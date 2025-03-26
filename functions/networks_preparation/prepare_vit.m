function [vit_net,vit_tl,vit_ft] = prepare_vit(INPUT_SIZE,OUTPUT_CLASSESS,tl_wlr,tl_blr,ft_wlr,ft_blr)
%prepare_vit This function prepares three different vision transformer.
   
% Prepare vit for the training from scratch
vit_net = visionTransformer;
cleanNetwork(vit_net); %clean all the weights

% Prepare vit for Transfer Learning
vit_tl = visionTransformer;

% Prepare vit for Fine Tuning
vit_ft = visionTransformer;

%For Vision Transformer, the size is fixed to 384x384 (the input layer does
%not change)

%Last fully connected layer
head_from_scratch = fullyConnectedLayer(OUTPUT_CLASSESS,Name="head");
head_layer_tl = fullyConnectedLayer(OUTPUT_CLASSESS,Name="head",WeightLearnRateFactor=tl_wlr,BiasLearnRateFactor=tl_blr);
head_layer_ft = fullyConnectedLayer(OUTPUT_CLASSESS,Name="head",WeightLearnRateFactor=ft_wlr,BiasLearnRateFactor=ft_blr);



vit_net=replaceLayer(vit_net,"head",head_from_scratch);   
vit_tl=replaceLayer(vit_tl,"head",head_layer_tl); 
vit_ft=dl_freezeNetwork(vit_ft,LayersToIgnore="SelfAttentionLayer"); %lock layers (for finetuning step)
vit_ft=replaceLayer(vit_ft,"head",head_layer_ft); %only fc6 and fc8 will learn




end

