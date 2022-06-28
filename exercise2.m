% Training of neural network for image reconstruction of digits propagated 
% through multimode fiber

clear all
close all
%% Genereal params for execution/Trainig

verbose = true; % if true : shows train process
validationFrequency = 200; % higher freq will speed script execution, lower could improve accuracy
epochs = 8; % number of epochs for training
miniBatchSize  = 128; % higher size will speed train and val process


%% Load training data
% load file "DATA_MMF_28.mat"
dataSet = load("DATA_MMF_16.mat");




%% Create Neural Network Layergraph MLP
% Layers = [];

xTrainDimensions = [size(dataSet.XTrain,1), size(dataSet.XTrain,1)];
yTrainDimensions = [size(dataSet.YTrain,1), size(dataSet.YTrain,1)];

Layers = [imageInputLayer([xTrainDimensions 1],"Name","Input")
    
fullyConnectedLayer(xTrainDimensions(1)^2,"Name","Fc1")

reluLayer("Name","Relu1")

fullyConnectedLayer(yTrainDimensions(1)^2,"Name","Fc2")

reluLayer("Name","Relu2")

depthToSpace2dLayer(yTrainDimensions,"Name","dts1")

regressionLayer("Name","Ouput")
];

%% Training network
% define "trainingOptions"
% training using "trainNetwork"
lr = 0.01;
options = trainingOptions('adam', ...
    'MiniBatchSize',miniBatchSize, ...
    'MaxEpochs',epochs, ...
    'InitialLearnRate',lr, ...
    'Shuffle','every-epoch', ...
    'ValidationData',{dataSet.XValid,dataSet.YValid}, ...
    'ValidationFrequency',validationFrequency, ...
    'Verbose',verbose);

% Training

[net, trainHistory] = trainNetwork(dataSet.XTrain, dataSet.YTrain, Layers, options);

%% Calculate Prediction 
% use command "predict"
Prediction = predict(net, dataSet.XTest);


%% Evaluate Network
% calculate RMSE, Correlation, SSIM, PSNR


% calc RMSE per prediction:
%% Boxplots for step 6 of instructions

%% Step 7: create Neural Network Layergraph U-Net
% Layers = [];

%% Boxplots for step 8 of instructions