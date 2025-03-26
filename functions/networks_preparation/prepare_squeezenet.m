function [squeeze_net,squeeze_tl,squeeze_ft] = prepare_squeezenet(INPUT_SIZE,OUTPUT_CLASSESS,tl_wlr,tl_blr,ft_wlr,ft_blr)
%PREPARE_SQUEEZE Summary of this function goes here
%   Detailed explanation goes here

% Prepare squeezenet for the training from scratch
squeeze_net = squeezenet("Weights","None");
cleanNetwork(squeeze_net);
% Prepare squeezenet for the fine tuning
squeeze_tl = layerGraph(squeezenet());
squeeze_ft = layerGraph(squeezenet());


image_input_layer = imageInputLayer(INPUT_SIZE,Name="data");

output_layer_from_scratch = convolution2dLayer([1 1],OUTPUT_CLASSESS,Name="conv10");
output_ft = convolution2dLayer([1 1],OUTPUT_CLASSESS,Name="conv10",WeightLearnRateFactor=tl_wlr,BiasLearnRateFactor=tl_blr);
output_tl = convolution2dLayer([1 1],OUTPUT_CLASSESS,Name="conv10",WeightLearnRateFactor=ft_wlr,BiasLearnRateFactor=ft_blr);


squeeze_net=replaceLayer(squeeze_net,"data",image_input_layer);
squeeze_net=replaceLayer(squeeze_net,"conv10",output_layer_from_scratch);

classification_tl = classificationLayer("Classes","auto");
squeeze_tl=replaceLayer(squeeze_tl,"data",image_input_layer);
squeeze_tl=replaceLayer(squeeze_tl,"conv10",output_tl);
squeeze_tl=replaceLayer(squeeze_tl,"ClassificationLayer_predictions",classification_tl);



classification_ft = classificationLayer("Classes","auto");
squeeze_ft=replaceLayer(squeeze_ft,"data",image_input_layer);
freezeNetwork(squeeze_ft);
squeeze_ft=replaceLayer(squeeze_ft,"conv10",output_ft);
squeeze_ft=replaceLayer(squeeze_ft,"ClassificationLayer_predictions",classification_ft);



end

