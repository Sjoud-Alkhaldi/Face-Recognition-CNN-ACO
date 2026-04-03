
%  1  load the dataset

function [im_Train, im_Validation, im_Test] = prepareDataset()

% Dataset path :
im_Path = fullfile(pwd, 'CelebA');

% Read images and labels :
imds = imageDatastore(im_Path, ...
    'IncludeSubfolders', true, ...
    'LabelSource', 'foldernames');

% Split data: 70% training :
[im_Train, imdsRest] = splitEachLabel(imds, 0.7, 'randomized');

% Split remaining into validation and testing :
[im_Validation, im_Test] = splitEachLabel(imdsRest, 0.5, 'randomized');

end
