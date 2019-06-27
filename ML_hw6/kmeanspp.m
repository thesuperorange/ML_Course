
function [center] =kmeanspp(X,k)
    center = X(randperm(size(X,1),1),:); %random pick 1 center
    L = ones(1,size(X,1));
    for i = 2:k
        D = X-center(L,:); %-1 
        D2=sqrt(dot(D,D,2));
        sum = cumsum(D2); % accumulate sum 
        sum_p = sum/sum(end);
        center(i,:) = X(find(rand < sum_p,1),:); %random a number in 0~1
    end

end