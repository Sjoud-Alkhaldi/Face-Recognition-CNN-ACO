function accuracy = trainACO_CNN(augTrain, augValidation, im_Train, im_Validation) %%

%% Train model
trainedNet = trainCNN_GoogLeNet(augTrain, augValidation, im_Train);

%% Extract features

features = activations(trainedNet, augTrain, 'pool5-7x7_s1', 'OutputAs', 'rows');
labels = im_Train.Labels;

%% ACO feature selection

bestFeatures = runACO(features, labels);

%% Train KNN
selectedData = features(:, logical(bestFeatures));
finalModel = fitcknn(selectedData, labels, 'NumNeighbors', 3);

%% Validation
valFeatures = activations(trainedNet, augValidation, 'pool5-7x7_s1', 'OutputAs', 'rows');
valLabels = im_Validation.Labels;

selectedVal = valFeatures(:, logical(bestFeatures));
predictedLabels = predict(finalModel, selectedVal);

accuracy = sum(predictedLabels == valLabels) / numel(valLabels);
disp(['Accuracy: ', num2str(accuracy * 100), '%']);

figure;
confusionchart(valLabels, predictedLabels);
title('Confusion Matrix');

save('FinalModel.mat', 'trainedNet', 'finalModel', 'bestFeatures');

end


%% ===== CNN Training =====
function trainedNet = trainCNN_GoogLeNet(augTrain, augValidation, im_Train)

net = googlenet;
lgraph = layerGraph(net);

numClasses = numel(categories(im_Train.Labels));

newFC = fullyConnectedLayer(numClasses, 'Name', 'fc_new', ...
    'WeightLearnRateFactor', 10, ...
    'BiasLearnRateFactor', 10);

newSoftmax = softmaxLayer('Name', 'softmax_new');
newClassLayer = classificationLayer('Name', 'output_new');

lgraph = replaceLayer(lgraph, 'loss3-classifier', newFC);
lgraph = replaceLayer(lgraph, 'prob', newSoftmax);
lgraph = replaceLayer(lgraph, 'output', newClassLayer);

options = trainingOptions('adam', ...
    'MiniBatchSize', 8, ...
    'MaxEpochs', 20, ...
    'InitialLearnRate', 1e-4, ...
    'L2Regularization', 0.05, ...
    'Shuffle', 'every-epoch', ...
    'ValidationData', augValidation, ...
    'Plots', 'training-progress', ...
    'Verbose', false);

trainedNet = trainNetwork(augTrain, lgraph, options);

end


%% ===== ACO =====
function bestFeatures = runACO(features, labels)

numFeatures = size(features, 2);

numAnts = 20;
numIterations = 10;
evaporationRate = 0.35;

pheromone = ones(1, numFeatures);
heuristic = var(features) + 0.01;

bestScore = 0;
bestFeatures = [];

for iter = 1:numIterations
    
    allSolutions = zeros(numAnts, numFeatures);
    scores = zeros(numAnts, 1);
    
    for ant = 1:numAnts
        
        selectedFeatures = zeros(1, numFeatures);
        
        for f = 1:numFeatures
            prob = pheromone(f) * heuristic(f);
            
            if rand < prob / (max(pheromone) * max(heuristic) + 0.1)
                selectedFeatures(f) = 1;
            end
        end
        
        if sum(selectedFeatures) == 0
            selectedFeatures(randi(numFeatures)) = 1;
        end
        
        selectedData = features(:, logical(selectedFeatures));
        mdl = fitcknn(selectedData, labels, 'NumNeighbors', 3);
        cvmdl = crossval(mdl, 'KFold', 3);
        score = 1 - kfoldLoss(cvmdl);
        
        allSolutions(ant, :) = selectedFeatures;
        scores(ant) = score;
        
        if score > bestScore
            bestScore = score;
            bestFeatures = selectedFeatures;
        end
    end
    
    pheromone = (1 - evaporationRate) * pheromone + sum(scores);
    
    disp(['Iteration ', num2str(iter), ' - Best: ', num2str(bestScore * 100)]);
end

end