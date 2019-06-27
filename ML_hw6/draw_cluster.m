
function draw_cluster(data,cluster)
[data_num,dim] = size(data);
     colorstring = 'rbgyk';
    for i = 1:data_num
        if(dim==2)
            plot(data(i,1),data(i,2),  colorstring(cluster(i)),'marker','.');
        elseif(dim>2)
            plot3(data(i,1),data(i,2),data(i,3),  colorstring(cluster(i)),'marker','.');
        end
        hold on;
    end
	
end
