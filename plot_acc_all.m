load accuracy/accuracy_Lt_4500/hunxiao_Lt_4500
acc = newAccAll;
n = -5:15;
figure()
hold on 
grid on
h1 = plot(n, acc(1,:), 'bo-');
set(h1,{'LineWidth'},{0.8});
h2=plot(n, acc(2,:), 'm<-');
set(h2,{'LineWidth'},{0.8});
h3=plot(n, acc(3,:), 'm^-');
set(h3,{'LineWidth'},{0.8});
h4=plot(n, acc(4,:), 'k*-');
set(h4,{'LineWidth'},{0.8});
h5=plot(n, acc(5,:), 'ks-');
set(h5,{'LineWidth'},{0.8});
h6=plot(n, acc(6,:), 'r+-');
set(h6,{'LineWidth'},{0.8});
h7=plot(n, acc(7,:), 'rd-');
set(h7,{'LineWidth'},{0.8});
h8=plot(n, acc(8,:), 'b>-');
set(h8,{'LineWidth'},{0.8});
xlim([-5,15])
%ylim([0.85,1])
xlabel('Composite SNR (dB)')
ylabel('{P_{cc}}')

legend('2ASK', '2FSK', '4FSK', 'BPSK', 'QPSK', '16QAM', '64QAM', 'MSK')
