

 function W=LDA_ml(X,label,k)
    [data_points,dims] = size(X);
    label_count = size(unique(label),1);
    total_mean = mean(X);
    %between class
    class_mean = zeros(label_count,dims);
    Sb=zeros(dims,dims);

    for i=1:label_count
         class_mean(i,:)=mean(X(label==i,:));
        class_count = size(X(label==i,:),1);
        sj = total_mean-class_mean(i,:);
        Sb = Sb+class_count*sj'*sj;
    end

    %within class
    Sw=zeros(dims,dims);
    for i=1:label_count
        Sw_tmp = X(label==i,:) - repmat(class_mean(i,:),[size(X(label==i,:),1) 1]);
        Sw=Sw+Sw_tmp'*Sw_tmp;
    end

    [V D] = eig(pinv(Sw)*Sb);
    variance =diag(D);
    [out,idx] = sort(variance,'descend');
    Vec = V(:,idx);
    W = Vec(:, 1:k);
 end

