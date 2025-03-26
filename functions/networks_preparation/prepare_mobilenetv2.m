function [mobilenetv2_net,mobilenetv2_tl,mobilenetv2_ft] = prepare_mobilenetv2(INPUT_SIZE,OUTPUT_CLASSESS,tl_wlr,tl_blr,ft_wlr,ft_blr)
%PREPARE_mobilenetv2 Summary of this function goes here
%   Detailed explanation goes here
 
mobilenetv2_net = mobilenetv2("Weights","none");
cleanNetwork(mobilenetv2_net); %clean all the weights

%generic Layer
image_input_layer = imageInputLayer(INPUT_SIZE,Name="input_1");
classification_layer_scratch = classificationLayer("Classes","auto");
classification_layer_tl = classificationLayer("Classes","auto");
classification_layer_ft = classificationLayer("Classes","auto");

% Prepare mobilenetv2net for the fine tuning
mobilenetv2_tl = layerGraph(mobilenetv2());
mobilenetv2_ft = layerGraph(mobilenetv2());


fc_layer_scratch = fullyConnectedLayer(OUTPUT_CLASSESS,Name="Logits");
fc_layer_tl = fullyConnectedLayer(OUTPUT_CLASSESS,Name="Logits",WeightLearnRateFactor=tl_wlr,BiasLearnRateFactor=tl_blr);
fc_layer_ft = fullyConnectedLayer(OUTPUT_CLASSESS,Name="Logits",WeightLearnRateFactor=ft_wlr,BiasLearnRateFactor=ft_blr);



mobilenetv2_net=replaceLayer(mobilenetv2_net,"input_1",image_input_layer);
mobilenetv2_net=replaceLayer(mobilenetv2_net,"Logits",fc_layer_scratch);
mobilenetv2_net=replaceLayer(mobilenetv2_net,"ClassificationLayer_Logits",classification_layer_scratch);


mobilenetv2_tl=replaceLayer(mobilenetv2_tl,"input_1",image_input_layer);
mobilenetv2_tl=replaceLayer(mobilenetv2_tl,"Logits",fc_layer_tl);
mobilenetv2_tl=replaceLayer(mobilenetv2_tl,"ClassificationLayer_Logits",classification_layer_tl);


mobilenetv2_ft=replaceLayer(mobilenetv2_ft,"input_1",image_input_layer);
mobilenetv2_ft=freezeNetwork(mobilenetv2_ft);
mobilenetv2_ft=replaceLayer(mobilenetv2_ft,"Logits",fc_layer_ft);
mobilenetv2_ft=replaceLayer(mobilenetv2_ft,"ClassificationLayer_Logits",classification_layer_ft);



end

