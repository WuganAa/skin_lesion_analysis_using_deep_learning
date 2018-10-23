clc
clear


tic
imds = imageDatastore('F:\\MATLAB\\Data', ...
    'IncludeSubfolders',true, ...
    'FileExtensions','.jpg', ... 
    'LabelSource','foldernames');
imds = shuffle(imds);%randomize the images
numLabelimds = countEachLabel(imds);%get the labels of images
[imdsTrain, imdsTest] = splitEachLabel(imds,...
                                        0.7,'randomized');  
Trainnum=length(imdsTrain.Labels);
Testnum=length(imdsTest.Labels);
                                    
inputSize = [146 146 3];
augimdsTrain = augmentedImageDatastore(inputSize,imdsTrain);
augimdsTest = augmentedImageDatastore(inputSize,imdsTest);
%adapt the size of training set and testing set to the suitable one
toc

%decide the initial weighs and biases
a_array = exp(10:-11/957:-1);
w1 = (randn(7,7,3));
w2 = (randn(7,7));
w3 = (randn(5,5));
w4 = (randn(5,5));
w5 = (randn(3,3));
w6 = (randn(3,3));
w1_copy=w1;
w2_copy=w2;
w3_copy=w3;
w4_copy=w4;
w5_copy=w5;
w6_copy=w6;
b1=0;
b2=0;
b3=0;
b4=0;
b5=0;
b6=0;

w6_store = zeros(Trainnum+1, 3, 3);
w6_store(1,1:3,1:3) = w6;
%apply SGD method to train the CNN net
for i=1:Trainnum
    a = a_array(i);
    [image,img_label] = readByIndex(augimdsTrain,i);
    if imdsTrain.Labels(i)=='benign'
        new_label=1;
    else
        new_label=0;
    end
    [w1,w2,w3,w4,w5,w6,b1,b2,b3,b4,b5,b6]=...
        SGD_2(im2double(cell2mat(image.input)),...
        new_label,a,w1,w2,w3,w4,w5,w6,b1,b2,b3,b4,b5,b6);
    i
    w6_store(i+1,1:3,1:3) = w6;
    toc
end

Pre_Label=zeros(Testnum);
T_benign=0;
T_malignant=0;
F_benign=0;
F_malignant=0;

%apply the testing set to validify the trained CNN net
for j=1:Testnum
    [image,img_label]=readByIndex(augimdsTest,j);
    if CNN_net_2(im2double(cell2mat(image.input)),...
            w1,w2,w3,w4,w5,w6,b1,b2,b3,b4,b5,b6)>0.5
        Pre_Label(j)=1;
        if imdsTest.Labels(j)=='benign'
            T_benign=T_benign+1;
        else
            F_benign=F_benign+1;
        end
    else
        Pre_Label(j)=0;
        if imdsTest.Labels(j)=='benign'
            F_benign=F_benign+1;
        else
            T_benign=T_benign+1;
        end
    end
    j
    toc
end

% %abtain the parameter to consider the performance of the CNN net in the
% %practice
% Accuracy = mean(YPred == imdsTest.Labels);
% Sensitivity = T_benign / (T_benign + F_malignant);
% Specificity = T_malignant / (T_malignant + F_benign);
% Balanced_accuracy = ((T_benign / (T_benign + F_benign)) ...
%             + (T_malignant / (T_malignant + F_malignant))) / 2;
% Precision = T_benign / (T_benign + F_benign);
% F_measure = 2 * Precision * Sensitivity / (Precision + Sensitivity);