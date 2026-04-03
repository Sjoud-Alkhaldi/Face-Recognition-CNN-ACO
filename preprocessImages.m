function [augTrain, augValidation, augTest] = preprocessImages(im_Train, im_Validation, im_Test)

% AlexNet input size
inputSize = [227 227 3];

% Data Augmentation (training only), random rotation and horizontal flip
augmenter = imageDataAugmenter('RandRotation', [-15 15], 'RandXReflection', true);

% Preprocess Training Data: Resize + Augmentation
augTrain = augmentedImageDatastore(inputSize, im_Train, 'DataAugmentation', augmenter);

% Preprocess Validation Data: Resize only
augValidation = augmentedImageDatastore(inputSize, im_Validation);

% Preprocess Test Data: Resize only
augTest = augmentedImageDatastore(inputSize, im_Test);

end