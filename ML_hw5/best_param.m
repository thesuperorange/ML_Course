%# read dataset
x=csvread('Plot_X.csv');
y=csvread('Plot_Y.csv');
%[bestC,bestG]=grid(x,y)
train_data=csvread('X_train.csv');
train_label=csvread('Y_train.csv');
[bestC,bestG]=grid(train_data,train_label)



%[dataClass, data] = libsvmread('./heart_scale');
function [bestc,bestg]=grid(x,y)

bestcv = 0;
for log2c = -1:3,
  for log2g = -4:1,
    cmd = ['-v 5 -s 0 -t 1 -h 0 -c ', num2str(2^log2c), ' -g ', num2str(2^log2g)];
    cv = svmtrain(y, x, cmd);
    if (cv > bestcv),
      bestcv = cv; bestc = 2^log2c; bestg = 2^log2g;
    end
    fprintf('%g %g %g (best c=%g, g=%g, rate=%g)\n', log2c, log2g, cv, bestc, bestg, bestcv);
  end
end
end