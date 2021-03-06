
clear all
close all
%% Genereal params for execution/Trainig

verbose = true; % if true : shows train process
validationFrequency = 200; % higher freq will speed script execution, lower could improve accuracy
epochs = 80; % number of epochs for training
miniBatchSize  = 128; % higher size will speed train and val process


%% Load training data
% load file "DATA_MMF_28.mat"
dataSet = load("DATA_MMF_16.mat");


xTrain = dataSet.XTrain;
yTrain = dataSet.XTrain;
xVal = dataSet.XValid;
yVal = dataSet.YValid;
xTest = dataSet.XTest;
yTest = dataSet.YTest;

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

[net, trainHistory] = trainNetwork(xTrain, yTrain, Layers, options);

%% Calculate Prediction 
% use command "predict"
prediction = predict(net, xTest);


%% Evaluate Network
% calculate RMSE, Correlation, SSIM, PSNR

rmses = zeros(siyze(yTest,4),1);
    corrCoefs = zeros(siyze(yTest,4),1);
    ssims = zeros(siyze(yTest,4),1);
    psnrs = zeros(siyze(yTest,4),1);
for i=1:size(yTest,4)
    
    rmses(i) = calcRmse(yTest, prediction);
    corrCoefs(i) = calcCorrCoef(yTest, prediction);
    ssims(i) = ssim(yTest, prediction);
    psnrs(i) = psnr(yTest, prediction);
end


%% Boxplots for step 6 of instructions

%% Step 7: create Neural Network Layergraph U-Net
% Layers = [];

%% Boxplots for step 8 of instructions