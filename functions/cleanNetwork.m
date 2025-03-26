function net = cleanNetwork(net)
% cleanNetwork = cleanNetwork(net) discard all weights of the pre-trained
% network
%


numLayers = numel(net.Layers);

for i = 1:numLayers
    layer = net.Layers(i);
    p = string(properties(net.Layers(i)));
    idx = ismember(p,"Weights");

    if any(idx)
        p = p(idx);
        numProperties = numel(p);
        for j = 1:numProperties
            layer.Weights=randn(size(layer.Weights))* 0.0001;
            
        end

        name = layer.Name;
        net = replaceLayer(net,name,layer);
    end


end
end
