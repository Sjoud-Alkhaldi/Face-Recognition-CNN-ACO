function evaluateModel(YPred, YTest)

% Compute Confusion Matrix
confMat = confusionmat(YTest, YPred);

% Accuracy
accuracy = sum(diag(confMat)) / sum(confMat(:));

% Precision, Recall, F1 (for each class, then take the average)
precision = diag(confMat) ./ sum(confMat, 2);
recall = diag(confMat) ./ sum(confMat, 1)';
f1 = 2 * (precision .* recall) ./ (precision + recall);

% Handle NaN (in case of division by zero)
precision(isnan(precision)) = 0;
recall(isnan(recall)) = 0;
f1(isnan(f1)) = 0;

% Average
avgPrecision = mean(precision);
avgRecall = mean(recall);
avgF1 = mean(f1);

% Display results
fprintf('Accuracy: %.2f%%\n', accuracy * 100);
fprintf('Precision: %.2f\n', avgPrecision);
fprintf('Recall: %.2f\n', avgRecall);
fprintf('F1-Score: %.2f\n', avgF1);

% Plot Confusion Matrix
figure;
confusionchart(confMat);
title('Confusion Matrix');

end