clear all
load acc

cm(:,9) = cm(:,9)/(sum(sum((cm(1:8,1:8))))+sum(cm(:,9)))*8;
%cm(9,:) = cm(9,:)/(sum(sum((cm(1:8,1:8))))+sum(cm(9,:)));
for i=1:8
    a=sum(cm(:,i));
    cm(:,i) = cm(:,i)/a;
end


imagesc(cm);

colorbar();
ylabel('Classification result','FontName', 'Times New Roman', 'FontSize', 13)
xlabel('Actual result','FontName', 'Times New Roman', 'FontSize', 13)
xticklabels({'2ASK','2FSK','4FSK','BPSK','QPSK','16QAM','64QAM','MSK','N/A'})
yticklabels({'2ASK','2FSK','4FSK','BPSK','QPSK','16QAM','64QAM','MSK','N/A'})


for i=1:9
    for j=1:9
        if i==j && cm(i,j)>=0.001
            text(i,j,num2str(cm(j,i),'%.3f'),'HorizontalAlignment',...
            'center','FontName', 'Times New Roman', 'FontSize', 12, 'color','black')
        elseif cm(i,j)>=0.001
            text(i,j,num2str(cm(j,i),'%.3f'),'HorizontalAlignment',...
            'center','FontName', 'Times New Roman', 'FontSize', 12, 'color','white')
        end
    end
end