# Trustworthiness in Digital Health: A Proposal for a Quantitative Reliability Indicator for Uncertainty-Aware CNNs

Dear reviewers, to reproduce our experiment, please refer to the following sections. Please ensure that pseudo-randomization initialization is set to "0" using rng(0) and rand(0) in each MLX file.

## Datasets downloads
The datasets used in this work are public. The first one is the "ISIC skin lesion dataset," and the second one is the "EyePAC dataset."  Please note that, for the ISIC dataset, copyright reasons require you to download the correct images yourself. The list of images used for ISIC is described in the file: https://github.com/CAISLab-Unisa/trustworthiness/blob/main/Datasets/ISIC_Image_Labels.txt. The file contains the filename for each "MALIGNANT" and "BENIGN" classes.
For the "eyepac dataset," please refer to https://www.kaggle.com/competitions/diabetic-retinopathy-detection to download the dataset.

## Dataset pre-processing 
As described in our submission in Sections Background and Method, to improve image quality and remove artifacts, we performed various pre-processing steps on the datasets. In particular, Contrast Enhancement, Histogram Equalization, and image segmentation, as well as hole erosion (only for the ISIC dataset). For convenience, we provide two different files: for ISIC, the pre-processing function is in https://github.com/CAISLab-Unisa/trustworthiness/blob/main/functions/otsu_he.m) For EyePACS, the pre-processing functions are located at (https://github.com/CAISLab-Unisa/trustworthiness/blob/main/functions/rgb9cer_improve.m). To rebuild the Training, validation, and test datasets, you only need to apply these functions to each image.

## Convolutional Neural Networks Setups and Training
As described in the methods section, we investigated the behaviors of 19 network architectures: AlexNet, GoogleNet, DenseNet201, Resnet18, MobilenetV2, SqueezeNet, ShuffleNet, Resnet50, Resnet101, NASNetMobile, EfficientNetB0, VGG16, VGG19, InceptionV3, Xception, InceptionResNetV2, DarkNet19, and DarkNet53. We are aware that the setup processing is time-intensive, as we have provided our pipeline in the file https://github.com/CAISLab-Unisa/trustworthiness/blob/main/Step1_Run_CNN_Trainings.mlx. Please note that to allow this script to work, it is necessary to download all the supporting functions contained in the folder "function" (https://github.com/CAISLab-Unisa/trustworthiness/tree/main/functions) in the leading directory where you save Step1_Run_CNN_Trainings.mlx. Also, please note that our pipeline is still able to reproduce the experimentation using Visual Transformer and nasetlarge. However, for the scope of our contribution, you will find these options disabled. To activate or deactivate a network structure, set the corresponding network name in the "0" array to either "1" or "exec_EXPS."
Please be aware that the contribution reports the results of the "from-the-scratch" experimentations. However, the script we provided can perform the same experimentations for "fine-tuned" and "transfer-learning" scenarios. Therefore, the "execute_experiment" function we provided contains commented lines (lines 46-57). If you want to test other functions, please be aware that you will need a GPU with at least 12GB of VRAM and CUDA compatibility. Also, be sure to reset gpu memory after each Training using the gpuDevice(..) function.
In order to perform different experimentation, you have to change only the following:
-	OUTPUT_CLASSESS variable value (2 for melanoma, 5 for RD);
-	the "dataset_base_path" (for convenience, you may save each dataset with a different name in the main folder where you will save STEP1 mlx);
-	the "experimentation_name" value. 
## Training outputs 
Each experimentation run will store the trained networks, confusion matrices, and Training, validation, and test performances in separate subfolders. The subfolder structure is the following:
/ (your root directory)
/ISIC (may contain pre-processed ISIC images)
/ISIC/Benign (must contain the images for the benign class)
/ISIC/Malignant (must contain the image for the malignant class)
/DR (may contain pre-processed EyePac Images)
/DR/Mild (same structure as for ISIC… one folder for each class)
/DR/Moderate
/DR/No_DR
/DR/Proliferate_DR
/DR/Severe
/Trained (the function execute_experiment will store in this folder each trained network, one folder for each network. Also, the TrainingPlot will be stored in this folder.)
/Results (the function execute_experiment will store in this folder the confusion matrices and the performance data for each network trained and validated, one network per folder)





