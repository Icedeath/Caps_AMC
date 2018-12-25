clear all
load hunxiao3000
acc_3 = newAver;
pf_3 = pfAll;
pm_3 = pmAll;
load hunxiao4500
acc_5 = newAver;
pf_5 = pfAll;
pm_5 = pmAll;

acc_AMPT = [0.5,0.625,0.728,0.82,0.88,0.92];
X = [0,3,6,9,12,15];

n= 0:15;

acc_A=interp1(X,acc_AMPT,n,'Spline');
figure()
hold on 
grid on
h1=plot(n, acc_A, 'bd-');
set(h1,{'LineWidth'},{1});
h2=plot(n, acc_3, 'mo-');
set(h2,{'LineWidth'},{1});
h3=plot(n, acc_5, 'r^-');
set(h3,{'LineWidth'},{1});
legend('AMPT-based AMC in [10]','\it{P_{cc}} \rmwithout \it{L^t}', '\it{P_{cc}} \rmwith \it{L^t}')
xlabel('Composite SNR (dB)')
ylabel('\it{P_{cc}}')
%%%%%%%%%%%%%%%%%%%%%%%%pf and pm%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%55
figure()
hold on
grid on
h4=plot(n, pf_3, 'm*-');
set(h4,{'LineWidth'},{0.8});
h5=plot(n, pf_5, 'ms-');
set(h5,{'LineWidth'},{0.8});
h6=plot(n, pm_3, 'b+-');
set(h6,{'LineWidth'},{0.8});
h7=plot(n, pm_5, 'bd-');
set(h7,{'LineWidth'},{0.8});

legend('\it{p_{f}} \rmwithout \it{L^t}', '\it{p_{f}} \rmwith \it{L^t}',...
    '\it{p_{m}} \rmwithout \it{L^t}', '\it{p_{m}} \rmwith \it{L^t}')
xlabel('Composite SNR (dB)')
ylabel('\it{p_{f}} \rmand \it{p_m}')