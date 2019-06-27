

 %origin is for drawing
% kmeans X=origin
% spectral X=laplacian
% kernel X=kernel
function [ cluster, center ] = kmeans_ml( k, X, origin, outputname,center) 

[data_num,dim]=size(X);
randIdx = randperm(data_num,k);
if nargin<5
    center = X(randIdx,:);
end



cluster = zeros(1,data_num);
cluster_pre = cluster;

iter = 0;
stop = false; 

%h=figure();
    h=figure('visible','off');
        %draw initial center (for kmeans++ comparison)
% for i=1:k
%     plot(center(i,1),center(i,2),'k*','MarkerSize',12);
%     hold on;
% end
while ~stop
    for i = 1:data_num
        % init distance array dist
        dist = zeros(1,k);
        % compute distance to each centroid
        for j=1:k
            dist(j) = norm(X(i,:)-center(j,:));
        end
        % find index of closest centroid (= find the cluster)
        [~, clusterP] = min(dist);
        cluster(i) = clusterP;   
        
    end
    center = zeros(k,dim);
    for j = 1:k
        center(j,:) = mean(X(cluster==j,:),1);
    end
    
     %draw   
    draw_cluster(origin,cluster);
        
    if cluster_pre==cluster
        stop = true;
    end
    cluster_pre = cluster; 
    iter = iter + 1;
    saveas(h,sprintf('%s%d.png',outputname,iter));

end

fprintf('iterations:%d \n',iter);
end



