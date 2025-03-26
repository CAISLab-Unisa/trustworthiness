function [shadow_ad] = rgb9cer_improve(data)
%RGB9CER_IMPROVE Summary of this function goes here
%   Detailed explanation goes here


 s_lab = rgb2lab(data);
 max_luminosity = 100;
 L = s_lab(:,:,1)/max_luminosity;
 shadow_ad = s_lab;
 shadow_ad(:,:,1) = adapthisteq(L)*max_luminosity;
 shadow_ad = lab2rgb(shadow_ad); 
 
end

