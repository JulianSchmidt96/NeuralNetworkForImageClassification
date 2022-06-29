%% Grep Dataset and perform train/val split

digitDatasetPath = fullfile(matlabroot,'toolbox','nnet','nndemos', ...
    'nndatasets','DigitDataset');
fullDataset = imageDatastore(digitDatasetPath, ...
    'IncludeSubfolders',true,'LabelSource','foldernames');

% train/val split
[imdsTrain, imdsVal] = splitEachLabel(fullDataset, 0.8, 'randomize');


%% Seting parameters for the entire Scipt

verbose = false; % if true : shows train process
val_freq = 200; % higher freq will speed script execution, lower could improve accuracy
epochs = 8; % number of epochs for training

%% Building a Network Architecture

Layers = [
    imageInputLayer([28 28 1],'Normalization','none','Name','input')
    
    fullyConnectedLayer(1000, 'Name','fullyConnected1')
    reluLayer('Name','relu1')
    
    fullyConnectedLayer(10,'Name','fullyConnected2')
    softmaxLayer('Name','softmax')
    
    classificationLayer('Name','classification')
   ];
LayerGraph = layerGraph(Layers);


if verbose == true
    
    plot(LayerGraph)
end

%% Build Array with Learning Rates and Array with different minibatche sizes

start_lr = 10^-6;
end_lr = 10^-1;

learningRates = decadeArray(start_lr, end_lr, 10);

minibatches = [10,30,60,100,150,200,256];

%% Build a structure to save results

optimizers = {'adam', 'sgdm'};

results = struct;
results.learningRatesACC = zeros(length(optimizers), length(learningRates));


%% Testing different optimizers at different learning rates 

for optim_idx = 1:length(optimizers)

    optimizer = optimizers{:,optim_idx};

    for lr_idx = 1:length(learningRates )
        learningRate = learningRates(lr_idx);

        trainOptions = trainingOptions(optimizer, ...
            'MaxEpochs',epochs, ...
            'InitialLearnRate',learningRate, ...
            'Verbose',verbose, ...
            'ValidationData',imdsVal, ...
            'ValidationFrequency',val_freq, ...
            'Verbose',verbose);

            [net, prediction ,acc ] = train_and_predict(Layers, trainOptions, imdsTrain, imdsVal,imdsVal.Labels);

            % Save acc and mse in results struct
            results.learningRatesACC(optim_idx,lr_idx) = acc;

    end
end

%% Find best Optimizer and Learn rate from before 



maximumACC = max(max(results.learningRatesACC)) ;

[bestOptimizer_idx,bestLearnRate_idx]=find(results.learningRatesACC==maximumACC);

bestOptimizer = optimizers{:,bestOptimizer_idx};




bestLearningRate = learningRates(bestLearnRate_idx);


if verbose == true
    fprintf('Best ACC: %f\n', maximumACC);
    fprintf('Best Optimizer: %s\n', bestOptimizer);
    fprintf('Best Learning Rate: %f\n', bestLearningRate);
end

%% Train again with the best learning rate and the best optimizer but for more epochs



bestTrainOptions = trainingOptions(bestOptimizer, ...
'MaxEpochs',3 * epochs, ...
'InitialLearnRate',bestLearningRate, ...
'Verbose',verbose, ...
'ValidationData',imdsVal, ...
'ValidationFrequency',val_freq, ...
'Verbose',verbose);

[net, prediction ,acc ,mse, rmse] = train_and_predict(Layers, bestTrainOptions, imdsTrain, imdsVal);


%% calculate the accuracy per digit

results.acc_per_digit = zeros(1,10);
for i=1:10
    results.acc_per_digit(1,i) = accPerValue(prediction,imdsVal.Labels,i);
end


%% Investigating the influence of minibatch size onto the training time

results.miniBatchTimes = zeros(1,length(minibatches));
results.miniBatchACC = zeros(1,length(minibatches));


for i = 1:length(minibatches)


    options = trainingOptions(bestOptimizer, ...
            'InitialLearnRate',bestLearningRate, ...
            'MaxEpochs',epochs, ... 
            'Shuffle','every-epoch', ...
            'ValidationData',imdsVal, ...
            'ValidationFrequency',val_freq, ...
            'MiniBatchSize',minibatches(i), ...
            'Verbose',verbose);

            tic
            [net, prediction ,acc ,mse, rmse] = train_and_predict(Layers, options, imdsTrain, imdsVal);


            results.miniBatchTimes(1,i)=toc;
            results.miniBatchACC(1,i)=acc;
            results.miniBatchMSE(1,i)=mse;
            results.miniBatchRMSE(1,i)=rmse;
            
        end

%% Plotting the results


% plot generation accuracy per digit

bar(results.acc_per_digit)
xlabel('digit')
ylabel('accuracy')
title('accuracy per digit')

saveas(gcf,[pwd '/plot/digitacc.png'])
% plot generation learnrate evaluation


plot(learningRates, results.learningRatesACC(1,:));
hold on
plot(learningRates, results.learningRatesACC(2,:));
set(gca, 'XScale', 'log');
xlabel('learn rate');
ylabel('accuracy');
legend('adam', 'sgdm');
title('comparing optimizers at diff learn rates');

saveas(gcf,[pwd '/plot/optim_lr.png'])

close

% plot generation minibatchsize evaluation


plot(minibatches, results.miniBatchACC(1,:));

xlabel('mini batch size');
ylabel('accuracy');
ylim([0 1])
title('minibatchsize vs accuracy');

saveas(gcf,[pwd '/plot/optim_mb.png'])

% plot generation time per minibatchsize

plot(minibatches, results.miniBatchTimes(1,:));

xlabel('mini batch size');
ylabel('time in s');
title('minibatchsize vs training time');

saveas(gcf,[pwd '/plot/time_mb.png'])

