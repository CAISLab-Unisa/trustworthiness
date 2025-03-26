function [] = save_training_plot(output_filename)
%SAVE_TRAINING_PLOT 


    currentfig = findall(groot, 'Tag', 'NNET_CNN_TRAININGPLOT_UIFIGURE');
    img = screencapture(0, 'Position', currentfig(1,1).Position);
    imwrite(img,output_filename);    
    clear img;
    clear currentfig;

end

