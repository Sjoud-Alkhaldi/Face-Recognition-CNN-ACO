function [augTrain, augValidation, augTest] = preprocessImages(im_Train, im_Validation, im_Test)

% GoogLeNet input size
inputSize = [224 224 3];

% Data Augmentation (training only)
augmenter = imageDataAugmenter( ...
    'RandRotation', [-30 30], ...
    'RandXReflection', true, ...
    'RandXTranslation', [-15 15], ...   
    'RandYTranslation', [-15 15]);

% Preprocess Training Data: Resize + Augmentation
augTrain = augmentedImageDatastore(inputSize, im_Train, 'DataAugmentation', augmenter);

% Preprocess Validation Data: Resize only
augValidation = augmentedImageDatastore(inputSize, im_Validation);

% Preprocess Test Data: Resize only
augTest = augmentedImageDatastore(inputSize, im_Test);

end