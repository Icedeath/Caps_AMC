clear all
load acc

cm(:,9) = cm(:,9)/(sum(sum((cm(1:8,1:8))))+sum(cm(:,9)));
cm(9,:) = cm(9,:)/(sum(sum((cm(1:8,1:8))))+sum(cm(9,:)));
for i=1:8
    a=sum(cm(:,i));
    cm(1:8,i) = cm(1:8,i)/a;
end


imagesc(cm);
colorbar();
ylabel('Classification result')
xlabel('Actual result')
