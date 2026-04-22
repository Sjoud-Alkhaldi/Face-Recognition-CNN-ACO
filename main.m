rng(42);

% 1. Load the model and variables
load('FinalModel.mat');

% 2. Prepare the images (to define im_Test)
[im_Train, im_Validation, im_Test] = prepareDataset();

% 3. Test model
[predictedLabels, trueLabels] = testModel(trainedNet, im_Test, finalModel, bestFeatures);

% 4. Evaluation
evaluateModel(predictedLabels, trueLabels);