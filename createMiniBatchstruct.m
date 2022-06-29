function miniBatches = createMiniBatchstruct(data, miniBatchSize)

    %{
        To-Do:

        miniBatches should contain numMiniBatches elements, each of which is a stuct with the following fields:
            images
            Labels

    %}
    
    totalDataLen = size(data.Labels, 1) ;
    numMiniBatches = miniBatchSize / totalDataLen;
    miniBatches = struct();

    for i = 1:numMiniBatches
        miniBatches.Labels = data.Labels(:, (i-1)*totalDataLen+1:i*totalDataLen);
        miniBatches.Features = data.Features(:, (i-1)*totalDataLen+1:i*totalDataLen);
    end
     
end