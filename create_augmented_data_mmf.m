%% Load non augmented training data
load('DATA_MMF_16.mat');            % data sets
load('MMF_Param_16.mat');           % MMF parameters
N = size(XTrain,4);                 % number of training images
r = size(XTrain,1);                 % resolution of training images=[r,r]

%% New training data
XTrain_aug = zeros(r,r,1,2*N);
YTrain_aug = zeros(r,r,1,2*N);
%%

tform1 = randomAffine2d('Rotation',[-45 45]);
tform2 = randomAffine2d('Rotation',[-35 35]);
%% Data Augmentation (2 new images per training image)
for i1=1:N
    original_image = XTrain(:,:,:,i1);
    
    % Data Augmentation 1
    aug_image =imwarp(original_image,tform1);
    
    [XTrain_aug(:,:,:,i1), YTrain_aug(:,:,:,i1)] = mmf(aug_image,r,M_T,modes_n);
    
    % Data Augmentation 2
    aug_image =imwarp(original_image,tform2);
    [XTrain_aug(:,:,:,N+i1), YTrain_aug(:,:,:,N+i1)] = mmf(aug_image,r,M_T,modes_n);
    
    disp([num2str(i1) '/' num2str(N)]);
end

%% Save Augmented Training Data
XTrain = cat(4,XTrain,XTrain_aug);
YTrain = cat(4,YTrain,YTrain_aug);
save('DATA_MMF_16_aug.mat','XTrain','YTrain','XValid','YValid','XTest','YTest');