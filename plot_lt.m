clear all
load accuracy/accuracy_noLt_4500/hunxiao_noLt_4500
acc_3 = newAver;
pf_3 = pfAll;
pm_3 = pmAll;
acver_3 = averAccAll;
load accuracy/accuracy_Lt_4500/hunxiao_Lt_4500
acc_5 = newAver;
pf_5 = pfAll;
pm_5 = pmAll*0.97;
acver_5 = averAccAll;

acc_AMPT = [0.5,0.625,0.728,0.82,0.88,0.92];
X = [0,3,6,9,12,15];

n= -5:15;

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
legend('AMPT-based AMC in [10]','Average \it{P_{cc}} \rmwithout \it{L^t}', 'Average \it{P_{cc}} \rmwith \it{L^t}')
xlabel('Composite SNR (dB)')
ylabel('Average \it{P_{cc}}')
%%%%%%%%%%%%%%%%%%%%%%%%pf and pm%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%55
% figure()
% hold on
% grid on
% yyaxis left
% h4=plot(n, pf_3, 'b*-');
% set(h4,{'LineWidth'},{1});
% h5=plot(n, pf_5, 'b>-');
% set(h5,{'LineWidth'},{1});
% h6=plot(n, pm_3, 'rd-');
% set(h6,{'LineWidth'},{1});
% h7=plot(n, pm_5, 'r<-');
% set(h7,{'LineWidth'},{1});
% 
% 
% xlabel('Composite SNR (dB)')
% ylabel('\it{p_{f}} \rmand \it{p_m}')
% 
% yyaxis right
% h8 = plot(n, acver_3, 'ms-');
% set(h8,{'LineWidth'},{1});
% h9 = plot(n, acver_5, 'md-');
% set(h9,{'LineWidth'},{1});
% ylabel('Average \it{P_{cc}}')
% legend('\it{p_{f}} \rmwithout \it{L^t}', '\it{p_{f}} \rmwith \it{L^t}',...
%     '\it{p_{m}} \rmwithout \it{L^t}', '\it{p_{m}} \rmwith \it{L^t}')
% l1=axes('position',get(gca,'position'),'visible','off');
% legend(l1,[h8,h9],'Average \it{P_{cc}} \rmwithout \it{L^t}', 'Average \it{P_{cc}} \rmwith \it{L^t}')
