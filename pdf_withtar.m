clear all
load final_output

tar = 2;
label = 0;
plot_pdf(tar,label,y_train,y_pred1,'bo-')
hold on
grid on

tar = 1;
label = 0;
plot_pdf(tar,label,y_train,y_pred1,'ro-')

tar = 2;
label = 1;
plot_pdf(tar,label,y_train,y_pred1,'k+-')
hold on
grid on

tar = 1;
label = 1;
plot_pdf(tar,label,y_train,y_pred1,'m+-')

legend('tar=2,label=0','tar=1,label=0','tar=2,label=1','tar=1,label=1')


function plot_pdf(tar, label, y_train, y_pred, var)
    y_idx = sum(y_train,2);
    y = y_pred(y_idx==tar,:);
    y_ref = y_train(y_idx==tar,:);
    y = y(y_ref==label);
    y = reshape(y, 1, size(y,1)*size(y,2));
    y_a = linspace(0,1,101);
    [f, xi] = ksdensity(y, y_a);
    plot(xi, f, var)
end
