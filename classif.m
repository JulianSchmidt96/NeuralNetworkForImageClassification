digitDatasetPath = fullfile(matlabroot,'toolbox','nnet','nndemos', ...
    'nndatasets','DigitDataset');
fullDataset = imageDatastore(digitDatasetPath, ...
    'IncludeSubfolders',true,'LabelSource','foldernames');
%%

[imdsTrain, imdsVal] = splitEachLabel(fullDataset, 0.8, 'randomize');
%% Set script parameter
verbose = false; % if true : shows train process
val_freq = 200; % higher freq qill peed script execution, lower could improve accuracy
%%

layers = [
    imageInputLayer([28 28 1],'Normalization','none','Name','input')
    
    fullyConnectedLayer(1000, 'Name','fullyConnected1')
    reluLayer('Name','relu1')
    
    fullyConnectedLayer(10,'Name','fullyConnected2')
    softmaxLayer('Name','softmax')
    
    classificationLayer('Name','classification')
   ];




start_lr = 10^-6;
end_lr = 10^-1;
minibatches = [10,30,60,100,150,200,256];

lrs = multiplicatedarray(start_lr, end_lr, 10);
results = struct;
results.lr = zeros(2,length(lrs));
results.mb = zeros(2,length(minibatches));
results.mbtimes = zeros(2,length(minibatches));
opts = {'adam','sgdm'};
optimizier = opts{1};
%%
for opt =1:length(opts)


    
    for lr = 1:length(lrs)

            options = trainingOptions(opts{:,opt}, ...
            'InitialLearnRate',lrs(lr), ...
            'MaxEpochs',8, ... 
            'Shuffle','every-epoch', ...
            'ValidationData',imdsVal, ...
            'ValidationFrequency',val_freq, ...
            'Verbose',verbose);


            net = trainNetwork(imdsTrain,layers,options);
            prediction =classify(net, imdsVal);
            yval = imdsVal.Labels;
            accuracy = sum(prediction==yval)/numel(yval);
            results.lr(opt,lr)=accuracy;

           
    end

end
display('finished training with dif lrs')
%%
[best_accc,lr_idx] = max(results.lr(1,:));
lr = lrs(lr_idx)

acc_per_digit=zeros(1,10);
optimizer = 'adam'

options = trainingOptions(optimizer, ...
            'InitialLearnRate',lr, ...
            'MaxEpochs',8, ... 
            'Shuffle','every-epoch', ...
            'ValidationData',imdsVal, ...
            'ValidationFrequency',200, ...
            'Verbose',verbose);

net = trainNetwork(imdsTrain,layers,options);
prediction =classify(net, imdsVal);
yval = imdsVal.Labels;

for i=1:10
    acc_per_digit(i) = acc_per_value(prediction,yval,i);
end



%%
for opt =1:length(opts)



tStart = tic;           % pair 2: tic
n = length(minibatches);
T = zeros(1,n);
[best_accc,lr_idx] = max(results.lr(opt,:));
lr = lrs(lr_idx)

    for i = 1:n

        options = trainingOptions(opts{:,opt}, ...
            'InitialLearnRate',lr, ...
            'MaxEpochs',8, ... 
            'Shuffle','every-epoch', ...
            'ValidationData',imdsVal, ...
            'ValidationFrequency',200, ...
            'MiniBatchSize',minibatches(i), ...
            'Verbose',verbose);
        tic
        net = trainNetwork(imdsTrain,layers,options);
            prediction =classify(net, imdsVal);
            yval = imdsVal.Labels;
            accuracy = sum(prediction==yval)/numel(yval);
            results.mb(opt,i)=accuracy
            results.mbtimes(opt,i)=toc;
        T(i)= toc;  % pair 1: toc
    end
end


%%


% plot generation accuracy per digit

bar(acc_per_digit)
xlabel('digit')
ylabel('accuracy')
title('accuracy per digit')

saveas(gcf,[pwd '/plot/digitacc.png'])
% plot generation learnrate evaluation


plot(lrs, results.lr(1,:));
hold on
plot(lrs, results.lr(2,:));
set(gca, 'XScale', 'log');
xlabel('learn rate');
ylabel('accuracy');
legend('adam', 'sgdm');
title('comparing optimizers at diff learn rates');

saveas(gcf,[pwd '/plot/optim_lr.png'])

close

% plot generation minibatchsize evaluation


plot(minibatches, results.mb(1,:));

xlabel('mini batch size');
ylabel('accuracy');
ylim([0 1])
title('minibatchsize vs accuracy');

saveas(gcf,[pwd '/plot/optim_mb.png'])

% plot generation time per minibatchsize

plot(minibatches, results.mbtimes(1,:));

xlabel('mini batch size');
ylabel('time in s');
title('minibatchsize vs training time');

saveas(gcf,[pwd '/plot/time_mb.png'])

