clc;
clear;

%% Load Data
    imds = imageDatastore('data', ...
        'IncludeSubfolders',true, ...
        'LabelSource','foldernames');
    imds = shuffle(imds);
    
    % Divide the data into training, validation and testing data sets. 
    % Use 60% for training, 20% for validation and 20% for testing.
    [imdsTrain,imdsValidation,imdsTest] = splitEachLabel(imds,0.6,0.2,'randomized');
    
    
    % Display some sample images.
    numTrainImages = numel(imdsTrain.Labels);
    idx = randperm(numTrainImages,16);
    figure
    for i = 1:16
        subplot(4,4,i)
        I = readimage(imdsTrain,idx(i));
        imshow(I)
    end
    
%% Load Pretrained Network
    net = alexnet;
    inputSize = net.Layers(1).InputSize;
    
    % Remove the last three layers of alexnet
    layersTransfer = net.Layers(1:end-3);
    
    % Add three new layers, fullyConnectedLayer, softmaxLayer,
    % classificationLayer 
    numClasses = numel(categories(imdsTrain.Labels));
    layers = [
        layersTransfer
        fullyConnectedLayer(numClasses,'WeightLearnRateFactor',20,'BiasLearnRateFactor',20)
        softmaxLayer
        classificationLayer];

%% Data augmentation    
    % Use an augmented image datastore to automatically resize the images. 
    % Specify additional augmentation operations to perform on the images.
    % Data augmentation helps prevent the network from overfitting 
    % and memorizing the exact details of the training images.    

    pixelRange = [-30 30];
    imageAugmenter = imageDataAugmenter( ...
        'RandXReflection',true, ...
        'RandXTranslation',pixelRange, ...
        'RandYTranslation',pixelRange);
    augimdsTrain = augmentedImageDatastore(inputSize(1:2),imdsTrain, ...
        'DataAugmentation',imageAugmenter);
    
    augimdsValidation = augmentedImageDatastore(inputSize(1:2),imdsValidation);
    augimdsTest = augmentedImageDatastore(inputSize(1:2),imdsTest);
    
%% Train Network
    options = trainingOptions('sgdm', ...
    'MiniBatchSize',10, ...
    'MaxEpochs',6, ...
    'InitialLearnRate',1e-4, ...
    'ValidationData',augimdsValidation, ...
    'ValidationFrequency',3, ...
    'ValidationPatience',Inf, ...
    'Verbose',false, ...
    'Plots','training-progress');
    
    netTransfer = trainNetwork(augimdsTrain,layers,options);
%% Classify Testing Images
    % Classify the testing images using the fine-tuned network, 
    % and calculate the classification accuracy.
    [YPred,scores] = classify(netTransfer,augimdaTest);
    T_benign = sum(YPred == imdsTest.Labels ... 
                    & imdsTest.Labels == "benign");
    T_malignant = sum(YPred == imdsTest.Labels ...
                    & imdsTest.Labels == "malignant");
    F_benign = sum(YPred ~= imdsTest.Labels ...
                    & imdsTest.Labels == "benign");
    F_malignant = sum(YPred ~= imdsTest.Labels ...
                    & imdsTest.Labels == "malignant");
    accuracy = mean(YPred == imdsTest.Labels);
    Sensitivity = T_benign / (T_benign + F_malignant);
    Specificity = T_malignant / (T_malignant + F_benign);
    Balanced_accuracy = ((T_benign / (T_benign + F_benign)) ...
                + (T_malignant / (T_malignant + F_malignant))) / 2;
    Precision = T_benign / (T_benign + F_benign);
    F_measure = 2 * Precision * Sensitivity / (Precision + Sensitivity);
    
    % Display four sample validation images with predicted labels 
    % and the predicted probabilities of the images having those labels.
    figure
    for i = 1:4
        subplot(2,2,i);
        I = readimage(imdsTest,idx(i));
        imshow(I);
        label = YPred(idx(i));
        label_true = imdsTest.Labels(idx(i));
        title("Actual Label:" + string(label_true) + ", " ...
            + string(label) + ", " ...
            + num2str(100*max(scores(idx(i),:)),3) + "%");
    end
        
    