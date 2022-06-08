function acc = acc_per_value(prediction, truth, value)

            idx = find(double(truth)==value);
            pred = prediction(idx);
            val = truth(idx);
            acc = sum(pred==val)/numel(val);
end
