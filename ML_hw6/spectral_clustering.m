
%draw(X,class);
function [class, L, U, value] = spectral_clustering(X,W, k, outputname,type)

degs = sum(W, 2);
D    = sparse(1:size(W, 1), 1:size(W, 2), degs);
% compute Laplacian
L = D - W;

[U, value] = eigs(L,k,0);
value = diag(value);
    if(type==1)
        class = kmeans_ml(k,U,X,outputname);             
    elseif (type==0)
        class = kmeans_ml(k,U,U,outputname);       
    end

end
