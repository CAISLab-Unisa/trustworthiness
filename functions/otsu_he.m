function [SEGMENTED] = otsu_he(data)
%OTSU_HE Summary of this function goes here
%   Detailed explanation goes here
 image = data;
 I = uint8(image);
 numColors = 2;
 L = imsegkmeans(I,numColors);
 B = labeloverlay(I,L);
 B = imfill(B,'holes');
 I = rgb2gray(I);
 [~,threshold] = edge(I,'sobel');
 fudgeFactor = 0.5;
 BWs = edge(I,'sobel',threshold * fudgeFactor);
 se90 = strel('line',5,90);
 se0 = strel('line',3,0);
 BWsdil = imdilate(BWs,[se90 se0]);
 BWdfill = imfill(BWsdil,'holes');
 BWnobord = imclearborder(BWdfill,4);
 seD = strel('diamond',2);
 BWfinal = imerode(BWnobord,seD);
 BWfinal = imerode(BWfinal,seD);
 SEGMENTED = (image.*BWfinal);
end

