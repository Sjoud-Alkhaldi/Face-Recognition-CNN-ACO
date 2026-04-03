function accuracy = trainACO_CNN(augTrain, augValidation, im_Train, im_Validation)

%% --- Train CNN ---
trainedNet = trainCNN(augTrain, augValidation, im_Train);

%% --- Extract Features ---
features = activations(trainedNet, augTrain, 'fc7', 'OutputAs', 'rows');
labels = im_Train.Labels;

%% --- Run ACO ---
bestFeatures = runACO(features, labels);

%% --- Train Final Classifier ---
selectedData = features(:, logical(bestFeatures));
finalModel = fitcknn(selectedData, labels, 'NumNeighbors', 3);

%% --- Validation ---
valFeatures = activations(trainedNet, augValidation, 'fc7', 'OutputAs', 'rows');
valLabels = im_Validation.Labels;

selectedVal = valFeatures(:, logical(bestFeatures));
predictedLabels = predict(finalModel, selectedVal);

accuracy = sum(predictedLabels == valLabels) / numel(valLabels);
disp(['Accuracy: ', num2str(accuracy * 100), '%']);

figure;
confusionchart(valLabels, predictedLabels);
title('CNN + ACO Confusion Matrix');

end


function trainedNet = trainCNN(augTrain, augValidation, im_Train)

% Load AlexNet
net = alexnet;

% Remove the last 3 layers
layersTransfer = net.Layers(1:end-3);

% Number of classes
numClasses = numel(categories(im_Train.Labels));

% Create new layers
layers = [
    layersTransfer
    fullyConnectedLayer(numClasses, 'Name', 'fc8_new')
    softmaxLayer('Name', 'softmax')
    classificationLayer('Name', 'classoutput')
];

% Training options
options = trainingOptions('adam', ...
    'MiniBatchSize', 8, ...
    'MaxEpochs', 10, ...
    'InitialLearnRate', 1e-4, ...
    'Shuffle', 'every-epoch', ...
    'ValidationData', augValidation, ...
    'ValidationFrequency', 5, ...
    'Verbose', false, ...
    'Plots', 'training-progress');

% Train CNN
trainedNet = trainNetwork(augTrain, layers, options);

end


function bestFeatures = runACO(features, labels)

numFeatures = size(features, 2);

% ACO Parameters
numAnts = 10;
numIterations = 20;
evaporationRate = 0.5;
alpha = 1;
beta = 2;

% Initialization
pheromone = ones(1, numFeatures);
heuristic = var(features);

bestScore = 0;
bestFeatures = [];

for iter = 1:numIterations
    
    allSolutions = zeros(numAnts, numFeatures);
    scores = zeros(numAnts, 1);
    
    for ant = 1:numAnts
        
        selectedFeatures = zeros(1, numFeatures);
        
        for f = 1:numFeatures
            prob = (pheromone(f)^alpha) * (heuristic(f)^beta);
            
            if rand < prob / (1 + prob)
                selectedFeatures(f) = 1;
            end
        end
        
        % Prevent empty selection
        if sum(selectedFeatures) == 0
            selectedFeatures(randi(numFeatures)) = 1;
        end
        
        % Evaluate selected features using KNN
        selectedData = features(:, logical(selectedFeatures));
        mdl = fitcknn(selectedData, labels, 'NumNeighbors', 3);
        cvmdl = crossval(mdl, 'KFold', 3);
        loss = kfoldLoss(cvmdl);
        
        score = 1 - loss;
        
        allSolutions(ant, :) = selectedFeatures;
        scores(ant) = score;
        
        if score > bestScore
            bestScore = score;
            bestFeatures = selectedFeatures;
        end
    end
    
    % Update pheromone
    pheromone = (1 - evaporationRate) * pheromone;
    
    for ant = 1:numAnts
        pheromone = pheromone + scores(ant) * allSolutions(ant, :);
    end
    
end

end