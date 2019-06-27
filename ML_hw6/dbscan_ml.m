
function [class,noise]=dbscan_ml(x,MinPts,epsilon);
    figure;
    [m,n]=size(x);
    D=pdist2(x,x);

    [m,n]=size(x);
    Cluster_num = 1;
    visited=zeros(m,1);
    noise = zeros(m,1);
    class = zeros(m,1);
    for i=1:m
        if ~visited(i) && ~noise(i)
            Neighbors=find(D(i,:)<=epsilon);
            if length(Neighbors)<MinPts       
                noise(i)=1;
                class(i)=0;
            else 
                noise(i)=0;
                class(Neighbors)=Cluster_num;
                [class,noise,visited] =ExpandCluster(x,Neighbors,Cluster_num,visited,noise,D,epsilon,MinPts,m,class);
                Cluster_num=Cluster_num+1; 
            end
        end
    end

    i1=find(noise==1);
    class(i1)=0;
    
end

function [class,noise,visited] = ExpandCluster(data,Neighbors,Cluster_num,visited,noise,D,epsilon,MinPts,m,class)
    wait_count=0;
    while ~isempty(Neighbors)
        visited(Neighbors(1))=1;
        i1=find(D(Neighbors(1),:)<=epsilon);
        tmp_idx = Neighbors(1);
        Neighbors(1)=[];
       
        if length(i1)>1
            class(i1)=Cluster_num;
            if length(i1)<MinPts
                noise(tmp_idx)=1;
                class(tmp_idx)=0;
                
                plot(data(tmp_idx,1),data(tmp_idx,2),'g*');
                hold on;
            else
                noise(tmp_idx)=0;
            end
            for j=1:length(i1)
                if visited(i1(j))==0
                    visited(i1(j))=1;
                    Neighbors=[Neighbors i1(j)];   
                    class(i1(j))=Cluster_num;
                    draw_cluster(data,class);

                end
            end
        end
         if(mod(wait_count,100)==0)
                    waitforbuttonpress;
         end
        wait_count=wait_count+1;
    end
end
