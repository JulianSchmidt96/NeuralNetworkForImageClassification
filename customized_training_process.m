%% Load Training Data & define class catalog & define input image size

% download from MNIST-home page or import dataset from MATLAB
% https://www.mathworks.com/help/deeplearning/ug/data-sets-for-deep-learning.html
% http://yann.lecun.com/exdb/mnist/

% Specify training and validation data
% Recommended naming >>>
% Train: dataset for training a neural network
% Test: dataset for test a trained neural network after training process
% Valid: dataset for test a trained neural network during training process
% X: input / for Classification: image
% Y: output / for Classification: label
% for example: XTrain, YTrain, XTest, YTest, XValid, YValid


%% define network (dlnet)
Layers = [
    imageInputLayer([28 28 1],'Normalization','none','Name','input')
    
    fullyConnectedLayer(1000, 'Name','fullyConnected1')
    reluLayer('Name','relu1')
    
    fullyConnectedLayer(10,'Name','fullyConnected2')
    softmaxLayer('Name','softmax')
    
    
   ];

% convert to a layer graph
lgraph = layerGraph(Layers);
% Create a dlnetwork object from the layer graph.
dlnet = dlnetwork(lgraph);
% visualize the neural network
%analyzeNetwork(dlnet)

%% Specify Training Options (define hyperparameters)

% miniBatchSize
% numEpochs
% learnRate 
% executionEnvironment
% numIterationsPerEpoch 

% training on CPU or GPU(if available);
% 'auto': Use a GPU if one is available. Otherwise, use the CPU.
% 'cpu' : Use the CPU
% 'gpu' : Use the GPU.
% 'multi-gpu' :Use multiple GPUs
% 'parallel :


%% Train neural network

% initialize the average gradients and squared average gradients
% averageGrad
% averageSqGrad

% "for-loop " for training

miniBatchSize = 120;
numEpochs = 10;
learnRate = 0.01;
numIterationsPerEpoch = 10;

digitDatasetPath = fullfile(matlabroot,'toolbox','nnet','nndemos', ...
    'nndatasets','DigitDataset');
fullDataset = imageDatastore(digitDatasetPath, ...
    'IncludeSubfolders',true,'LabelSource','foldernames');

% train/val split
[imdsTrain, imdsVal] = splitEachLabel(fullDataset, 0.8, 'randomize');


% take slices from imdsTrain and store in array





data = imdsTrain;
miniBatches = createMiniBatchstruct(data, miniBatchSize)



for epoch = 1:numEpochs
    
   % updae learnable parameters based on mini-batch of data
    for i = 1:numIterationsPerEpoch
        
        % Read mini-batch of data and convert the labels to dummy variables.
        
        [X,Y] = next(mbq);

        


        % Convert mini-batch of data to a dlarray.
        X = dlarray(X');
        T = dlarray(T);
        
        
        
        % Evaluate the model gradients and loss using dlfeval and the
        % modelGradients helper function.

        [loss,gradients,state] = dlfeval(@modelmodelGradientsLoss,net,X,T);
        net.State = state;
        
        
        % Update the network parameters using the optimizer, like SGD, Adam
        [net,velocity] = sgdmupdate(net,gradients,velocity,learnRate,momentum);
        
        
        % Calculate accuracy & show the training progress.
        D = duration(0,0,toc(start),Format="hh:mm:ss");
        loss = double(loss);
        addpoints(lineLossTrain,iteration,loss)
        title("Epoch: " + epoch + ", Elapsed: " + string(D))
        drawnow
        % option: validation

    end
end


%% test neural network & visualization 

