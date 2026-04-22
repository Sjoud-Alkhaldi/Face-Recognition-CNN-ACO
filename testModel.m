function [predictedLabels, trueLabels] = testModel(trainedNet, im_Test, finalModel, bestFeatures)

    % Resize images automatically to 224x224(GoogLeNet)
    inputSize = [224 224];
    augTest = augmentedImageDatastore(inputSize, im_Test);

    % Extract features
    fprintf('Extracting features...\n');
    rawFeatures = activations(trainedNet, augTest, 'pool5-7x7_s1', 'OutputAs', 'rows');
    
    % Select features based on ACO
    selectedFeatures = rawFeatures(:, logical(bestFeatures));
    
    % Predict labels using KNN & Get actual labels
    predictedLabels = predict(finalModel, selectedFeatures);
    trueLabels = im_Test.Labels;
    
    % Show results
    disp(table(trueLabels(1:5), predictedLabels(1:5), 'VariableNames', {'Actual','Predicted'}));
end