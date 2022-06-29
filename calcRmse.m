function rmse= calcRmse(truth,prediction)
%{
_summary_
    Args:
        truth (array): can be a vector or matrix representing the true values
        prediction ( array): can be a vector or matrix representing
        predicted values<
    Returns:
        RMSE (float): RMSE

_description_
        This fct will calculate the RMSE beetween to arrays

%}

if (size(truth,4) ~= size(prediction,4))

    display("truth and prediction dimensions don't match")
    rmse = false
else
    rmse = sqrt(mean(truth-prediction))


end

