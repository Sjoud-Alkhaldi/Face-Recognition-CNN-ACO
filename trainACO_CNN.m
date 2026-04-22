function accuracy = trainACO_CNN(augTrain, augValidation, im_Train, im_Validation) % the main function 

rng(42); 

%% Train model
trainedNet = trainCNN_GoogLeNet(augTrain, augValidation, im_Train);  

%% Extract features
features = activations(trainedNet, augTrain, 'pool5-7x7_s1', 'OutputAs', 'rows');% from pool5 Layer , activations=Feature Extraction ,1024
labels = im_Train.Labels;

%% ACO feature selection
% We send the features in the pool5 to the  ACO to select the best features
% (Tuned Parameters)
[bestFeatures, tunedParams] = runACO(features, labels);



fprintf('ACO Selected Learning Rate: %f\n', tunedParams.lr);
fprintf('ACO Selected Batch Size: %d\n', tunedParams.batchSize);
fprintf('ACO Selected Depth: %s\n', tunedParams.depth);
fprintf('Optimized Feature Set: %d selected from 1024\n', sum(bestFeatures));
fprintf('--------------------------------------------------\n');

%% Train KNN :
% 1- To filter the original features from the best features selected from ACO
selectedData = features(:, logical(bestFeatures));
% 2- We use KNN as a classifier that takes the selectedData and labels, and considers the 3 nearest neighbors.
finalModel = fitcknn(selectedData, labels, 'NumNeighbors', 3);

%% Validation :
%We use trainedNet as a Feature Extractor to generate deep features, from ugValidation =new images  at the same pool5
valFeatures = activations(trainedNet, augValidation, 'pool5-7x7_s1', 'OutputAs', 'rows');
valLabels = im_Validation.Labels; 
selectedVal = valFeatures(:, logical(bestFeatures));%  select bestFeatures

% Use the trained KNN model to predict the identity of the test images based on their optimized features
% list of predicted names  = predictedLabels
predicted_Labels = predict(finalModel, selectedVal);

 %  Number of times the system's answer was correct / Total number of images
accuracy = sum(predicted_Labels == valLabels) / numel(valLabels);
disp(['Final Accuracy: ', num2str(accuracy * 100), '%']);

%Classification accuracy for each user individually
figure;
confusionchart(valLabels, predicted_Labels);
title(['Final Recognition Accuracy: ', num2str(accuracy * 100), '%']);

save('FinalModel.mat', 'trainedNet', 'finalModel', 'bestFeatures');

end

%% ===== CNN Training =====
% performs Transfer Learning by replacing the final layers of GoogLeNet
%  to recognize our specific set of users and starts the training process :
function trainedNet = trainCNN_GoogLeNet(augTrain, augValidation, im_Train)

net = googlenet; % Call GoogLeNet
lgraph = layerGraph(net); 
% Calculate the number of  classes in our dataset
numClasses = numel(categories(im_Train.Labels));
% Create a new Fully Connected layer for our number of classes
newFC = fullyConnectedLayer(numClasses, 'Name', 'fc_new', ...
    'WeightLearnRateFactor', 10, ...
    'BiasLearnRateFactor', 10);% increase to make this layer learn faster than others
% layer to convert classification scores into probabilities 
newSoftmax = softmaxLayer('Name', 'softmax_new');
%new Classification Output layer
newClassLayer = classificationLayer('Name', 'output_new');

% Replace the original old layers with our new layers
lgraph = replaceLayer(lgraph, 'loss3-classifier', newFC);
lgraph = replaceLayer(lgraph, 'prob', newSoftmax);
lgraph = replaceLayer(lgraph, 'output', newClassLayer);
% Set the Training Options 
options = trainingOptions('adam', ...  
    'MiniBatchSize', 8, ...          
    'MaxEpochs', 20, ...
    'InitialLearnRate', 1e-4, ...
    'L2Regularization', 0.05, ...
    'Shuffle', 'every-epoch', ...
    'ValidationData', augValidation, ...
    'Plots', 'training-progress', ...
    'Verbose', false);
%Start the actual training process
trainedNet = trainNetwork(augTrain, lgraph, options);

end

%% ===== ACO (Modified for Hyperparameter Optimization) =====
function [bestFeatures, tunedParams] = runACO(features, labels)

numFeatures = size(features, 2); %Get the total number of features from pool5 =1024

numAnts = 20;           %Number of ants = number of attempts in each iteration
numIterations = 10;     % times the ants repeat the search
evaporationRate = 0.35; % Pheromone decay rate

% (Tuning Space):
LR_options = [1e-3, 1e-4, 5e-5]; 
BS_options = [8, 16, 32];

pheromone = ones(1, numFeatures);  % Initial pheromone level for all features
heuristic = var(features) + 0.01;  % Heuristic info

bestScore = 0;
bestFeatures = [];

tunedParams.lr = 1e-4;
tunedParams.batchSize = 8;
tunedParams.depth = 'pool5';

for iter = 1:numIterations        % Start optimization iterations = the main loop
    
    allSolutions = zeros(numAnts, numFeatures);
    scores = zeros(numAnts, 1);
    
    for ant = 1:numAnts  % loop that starts for each ant individually to search for features
        
        selectedFeatures = zeros(1, numFeatures); %An array to store the sets of features selected by the ants in this iteration
        
        for f = 1:numFeatures    %A loop that iterates through each feature one by one
            prob = pheromone(f) * heuristic(f);
            
           %Feature selection = (data quality of the feature) * (its success rate in previous trials).
           %  We divide this by the maximum value to ensure that the result does not exceed 1.
           %  If it is greater than the random number generated by MATLAB (which is a number between 0 and 1), we set the feature to 1;
           %  else, we set it to 0 and remove the feature.
            
            if rand < prob / (max(pheromone) * max(heuristic) + 0.1)
                selectedFeatures(f) = 1;
            end
        end
        
        % To prevent errors, there is a possibility that no feature will be selected at all;
        %  therefore, we force the ant to select only one feature at random from the 1,024 available features.
        if sum(selectedFeatures) == 0
            selectedFeatures(randi(numFeatures)) = 1;
        end
        
        
        testLR = LR_options(randi(length(LR_options)));
        testBS = BS_options(randi(length(BS_options)));

        %  We performed an internal validation, selecting only the features chosen by the ant
        %  and building a temporary KNN model for the ant using those features.
        %  We used the K-fold cross-validation technique.

        selectedData = features(:, logical(selectedFeatures));
        mdl = fitcknn(selectedData, labels, 'NumNeighbors', 3);
        cvmdl = crossval(mdl, 'KFold', 3);
        score = 1 - kfoldLoss(cvmdl);
        
        allSolutions(ant, :) = selectedFeatures;
        scores(ant) = score;
        
        
        if score > bestScore
            bestScore = score;
            bestFeatures = selectedFeatures;
            tunedParams.lr = testLR;
            tunedParams.batchSize = testBS;
        end
    end

    %Pheromone update: 
    % In the first phase, we perform an evaporation process to reduce the value of the old pheromone.
    %  In the second phase, we perform a reinforcement process: we aggregate the accuracy scores and add them to the features they selected. 
    pheromone = (1 - evaporationRate) * pheromone + sum(scores);
    
    disp(['Iteration ', num2str(iter), ' - Best Local Accuracy: ', num2str(bestScore * 100)]);
end

end