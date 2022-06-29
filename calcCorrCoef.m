function [corrCoef] = calcCorrCoef(truth,prediction)
%{
  _summary_
    Args:
        truth (array): can be a vector or matrix representing the true values
        prediction ( array): can be a vector or matrix representing predicted values
    Returns:
        corrCoef (float): correlation coefficient

_description_
        This fct will calculate the correlation coefficient beetween to arrays


%}
corrCoefMatrix = corrcoef(truth, prediction);

corrCoef = corrCoefMatrix(1,2);
end

