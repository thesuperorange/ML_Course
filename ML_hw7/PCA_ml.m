

%norm_data = (X_PCA - min(X_PCA(:))) / ( max(X_PCA(:)) - min(X_PCA(:)) );
%img = reshape(X_PCA(1,:),28,28);
%fname = sprintf('face%d.jpeg', 1);
%imwrite(img,fname,'JPEG');


function W = PCA_ml(X,k)
    [data_pt, dims] = size(X);
    X = X - repmat(mean(X),[data_pt 1]);
    C=X'*X;
    [V, D] = eig(C);
    variance =diag(D);
    [out,idx] = sort(variance,'descend');
    Vec = V(:,idx);
    W = Vec(:, 1:k);
end