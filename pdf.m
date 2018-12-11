clear all
load final_output_noLT
figure()
label = 0;
x=plot_pdf(label,y_train,y_pred1,'r');


hold on
grid on


label = 1;
x=plot_pdf(label,y_train,y_pred1,'b^');


legend('{\nu_0}','{\nu_1}')
xlabel('a')
ylabel('{N^U}')

axes('position', [0.25,0.4,0.6,0.3])
label = 0;
x=plot_pdf(label,y_train,y_pred1,'r');
hold on
grid on
label = 1;
x=plot_pdf(label,y_train,y_pred1,'b^');
xlim([0.4,0.8])
ylim([0,0.0015])






function x=plot_pdf(label, y_train, y_pred, str)
    y1 = y_pred(y_train==label);
    n = size(y1,1)*size(y1,2);
    y1 = reshape(y1, 1, n);
    [y,x] = hist(y1, 99);
    y = y/size(y_train,1)/10;
    stem(x,y,str)
end
