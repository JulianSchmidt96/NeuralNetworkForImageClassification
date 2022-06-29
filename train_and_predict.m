function [net, prediction, accuracy,mse,rmse] = train_and_predict(layers, options, train_data, val_data )
    %{
    _summary_
        Args:
            layers: a cell array of layer definitions
            options: a struct of options as training parameters
            train_data (struct): a struct of training data, containing input and target data
            val_data (struct): a struct of validation data, containing input and target data
        Returns:
            net : a trained neural network
            prediction (array) : prediction of the trained network
            accuracy (float) : accuracy of the trained networks prediction
            mse (float) : mean squared error of the trained networks prediction
    _description
    This function trains a neural network and returns a prediction using the validation data as a test set.
    The function also returns the accuracy of the prediction and the mse of the prediction.
    %}
    net = trainNetwork(train_data, layers, options);
    prediction = classify(net, val_data);
    accuracy = sum(prediction == val_data.Labels)/numel(val_data.Labels);
    mse = mean((prediction == val_data.Labels).^2);
    rmse = sqrt(mse);

end
