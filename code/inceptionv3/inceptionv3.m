clc
clear

%% Load Data
    imds = imageDatastore('F:\\MATLAB\\Data', ...
        'IncludeSubfolders',true, ...
        'FileExtensions','.jpg', ... 
        'LabelSource','foldernames');
    imds = shuffle(imds);
    numLabelimds = countEachLabel(imds);

    % Divide the data into training, validation and testing data sets. 
    % Use 60% for training, 20% for validation and 20% for testing.
    [imdsTrain, imdsValidation, imdsTest] = splitEachLabel(imds,...
                                            0.6, 0.2,'randomized');

    % Display some sample images.
    numTrainImages = numel(imdsTrain.Labels);
    idx = randperm(numTrainImages,9);
    figure
    for i = 1:9
        subplot(3,3,i)
        I = readimage(imdsTrain,idx(i));
        imshow(I)
        label = imdsTrain.Labels(idx(i));
        title(string(label));
    end

%% Load Pretrained Network
    net = inceptionv3;
    
    % Extract the layer graph from the trained network
    lgraph = layerGraph(net);
    analyzeNetwork(lgraph)
    % Plot the layer graph.
    figure('Units','normalized','Position',[0.1 0.1 0.8 0.8]);
    plot(lgraph)
    
    % The first element of the Layers is the image input layer.
    inputSize = net.Layers(1).InputSize;
    
%% Replace Final Layers
    % To retrain, replace the last three layers of the network, which 
    % contain information on how to combine the features 
    % that the network extracts into class probabilities and labels. 
    lgraph = removeLayers(lgraph, {'predictions',...
                                    'predictions_softmax',...
                                    'ClassificationLayer_predictions'});
    % Add three new layers to the layer graph
    numClasses = numel(categories(imdsTrain.Labels));
    newLayers = [
        fullyConnectedLayer(numClasses,'Name','fc',...
                            'WeightLearnRateFactor',10,...
                            'BiasLearnRateFactor',10)
        softmaxLayer('Name','softmax')
        classificationLayer('Name','classoutput')];
    lgraph = addLayers(lgraph,newLayers);
    
    % Connect the last transferred layer remaining in the network 
    % to the new layers
    lgraph = connectLayers(lgraph,'avg_pool','fc');
    figure('Units','normalized','Position',[0.3 0.3 0.4 0.4]);
    plot(lgraph)
    ylim([0,10])
    
%% Freeze Initial Layers
    % Extract the layers and connections of the layer graph 
    % and select which layers to freeze.
    % Use the freezeWeights to set the learning rates to zero 
    % for the first several layers.
    % Use the createLgraphUsingConnections to reconnect all the layers 
    % in the original order. 
    layers = lgraph.Layers;
    connections = lgraph.Connections;

    layers(1:280) = freezeWeights(layers(1:280));
    lgraph = createLgraphUsingConnections(layers,connections);

%% Data augmentation
    % Use an augmented image datastore to automatically resize the images. 
    % Specify additional augmentation operations to perform on the images.
    % Data augmentation helps prevent the network from overfitting 
    % and memorizing the exact details of the training images.
    pixelRange = [-30 30];
    imageAugmenter = imageDataAugmenter( ...
        'RandXReflection',true, ...
        'RandYReflection',true, ...
        'RandXTranslation',pixelRange, ...
        'RandYTranslation',pixelRange);
    augimdsTrain = augmentedImageDatastore(inputSize(1:2),imdsTrain, ...
        'DataAugmentation',imageAugmenter);
    augimdsValidation = augmentedImageDatastore(inputSize(1:2),imdsValidation);
    augimdaTest = augmentedImageDatastore(inputSize(1:2),imdsTest);

%% Train Network
    % Specify the training options. 
    % Set InitialLearnRate to a small value to slow down learning 
    % in the transferred layers that are not already frozen.
    % In the previous step, increasing the learning rate factors 
    % for the fully connected layer to speed up learning 
    % in the new final layers. 
    % This combination of learning rate settings results in 
    % fast learning in the new layers, 
    % slower learning in the middle layers, 
    % and no learning in the earlier, frozen layers.
    options = trainingOptions('sgdm', ...
        'MiniBatchSize',10, ...
        'MaxEpochs',6, ...
        'Shuffle','every-epoch', ...
        'InitialLearnRate',1e-4, ...
        'ValidationData',augimdsValidation, ...
        'ValidationFrequency',3, ...
        'ValidationPatience',Inf, ...
        'Verbose',1 , ...
        'VerboseFrequency', 100, ...
        'Plots','training-progress');
    net = trainNetwork(augimdsTrain,lgraph,options);

%% Classify Testing Images
    % Classify the testing images using the fine-tuned network, 
    % and calculate the classification accuracy.
    [YPred,probs] = classify(net,augimdaTest);
    T_benign = sum(YPred == imdsTest.Labels ... 
                    & imdsTest.Labels == "benign");
    T_malignant = sum(YPred == imdsTest.Labels ...
                    & imdsTest.Labels == "malignant");
    F_benign = sum(YPred ~= imdsTest.Labels ...
                    & imdsTest.Labels == "benign");
    F_malignant = sum(YPred ~= imdsTest.Labels ...
                    & imdsTest.Labels == "malignant");
    Accuracy = mean(YPred == imdsTest.Labels);
    Sensitivity = T_benign / (T_benign + F_malignant);
    Specificity = T_malignant / (T_malignant + F_benign);
    Balanced_accuracy = ((T_benign / (T_benign + F_benign)) ...
                + (T_malignant / (T_malignant + F_malignant))) / 2;
    Precision = T_benign / (T_benign + F_benign);
    F_measure = 2 * Precision * Sensitivity / (Precision + Sensitivity);
    
    % Display four sample validation images with predicted labels 
    % and the predicted probabilities of the images having those labels.
    idx = randperm(numel(imdsTest.Files),4);
    figure
    for i = 1:4
        subplot(2,2,i)
        I = readimage(imdsTest,idx(i));
        imshow(I)
        label = YPred(idx(i));
        label_true = imdsTest.Labels(idx(i));
        title("Actual Label:" + string(label_true) + ", " ...
            + string(label) + ", " ...
            + num2str(100*max(probs(idx(i),:)),3) + "%");
    end