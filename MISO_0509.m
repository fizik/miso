%% ================================================================
%  10GBASE‑T   Legacy (无预补偿)  vs.  Proposed (MISO 预补偿 + FFE)
%  ‑‑ 全链路 Monte‑Carlo 计算 FEXT Suppression / DP‑SNR / BER  ‑‑
% ================================================================

%% ---------- 参数区 ----------
Nsym     = 3e4;      % 符号数（训练 + 验证）
trainSym = 4e3;      % 训练长度
SNR_dB   = 20;       % AWGN 信噪比 (调大→噪声小；18–25 dB 均可)
P_FEXTdB = -25;      % 单根 Cat‑6A 55 m 典型 ELFEXT (-25 dB 更具挑战)
tapFFE   = 3;        % 接收端 FFE tap 个数
muTx     = 2e-3;     % Tx 端 LMS 步长
muRx     = 5e-3;     % Rx 端 LMS 步长
rng(42);             % 固定随机种子，方便复现
%% ---------------------------------------------------------------

M     = 16;                                 % PAM‑16 (DSQ128 一维等价)
pamSym= @(n) (randi([-(M/2-1) (M/2-1)], n,1)*2+1);   % 生成 PAM16
sigma = 10^(-SNR_dB/20);                    % AWGN σ
alpha = 10^(P_FEXTdB/20);                   % FEXT 线性幅度

%% ---------- 4×4 通道矩阵 ----------
Hdir       = eye(4);                        % 直接通道
Hfext      = alpha * (randn(4) - 0.5);      % 纯实随机耦合
Hfext(1:5:end) = 0;                         % 对角清零
H          = Hdir + Hfext;                  % 总 4×4

%% ---------- 生成发送符号 ----------
txSym = pamSym(Nsym*4);  tx = reshape(txSym, Nsym,4);   % Nsym×4

%% ---------- Legacy：无预补偿 ----------
noiseL = sigma*randn(Nsym,4);
rxLegacy = tx*H.' + noiseL;                 % 直通 + 噪声

%% ---------- Proposed：MISO 预补偿 + FFE ----------
% 初始 MISO 系数：直接取 Hfext(1,2:4)
Lpre = zeros(1,4);    Lpre(2:4) = Hfext(1,2:4);
% Rx FFE 系数
Wffe = zeros(1,tapFFE);
yHist = zeros(1,tapFFE-1);      % 历史寄存器
z_hat = zeros(Nsym,1);          % 保存等化输出

for k = 1:Nsym
    %------ 发送端预补偿 (仅对 pair‑1 做) ------
    u = tx(k,:);                                      % 4‑元素
    u(1) = tx(k,1) - tx(k,2:4)*Lpre(2:4).';           % MISO 线性补偿
    
    %------ 通道 ------
    y = u * H.' + sigma*randn(1,4);                   % Rx 实信号
    yi = y(1);                                        % 关注第 1 对
    
    %------ Rx FFE ------
    yvec = [yi yHist];                                % tapFFE‑维
    zi   = Wffe * yvec.';                             % FFE 输出
    d    = tx(k,1);                                   % 训练目标
    err  = d - zi;
    
    %------ LMS 更新 (训练阶段) ------
    if k <= trainSym
        Wffe = Wffe + muRx * err * yvec;
        Lpre(2:4) = Lpre(2:4) - muTx * err * tx(k,2:4);
    end
    
    %------ 更新寄存器 & 保存输出 ------
    yHist = [yi yHist(1:end-1)];
    z_hat(k) = zi;
end

%% ---------- 统计指标 ----------
% 1) FEXT 抑制量  = 原始 FEXT 功率 / (剩余 FEXT 功率)
origFEXT   = tx(:,2:4)*Hfext(1,2:4).';               % Legacy FEXT 分量
predFEXT   = tx(:,2:4)*Lpre(2:4).';                  % 预补偿抵消量
resiFEXT   = origFEXT - predFEXT;                    % 剩余
FEXTsupp_dB= 10*(1 + log10( mean(origFEXT.^2) / mean(resiFEXT.^2) ) );

% 2) DP‑SNR (从验证区段计算)
testIdx  = trainSym+1 : Nsym;
dpSNR_L  = 10*log10( var(tx(testIdx,1)) / var(rxLegacy(testIdx,1)-tx(testIdx,1)) );
dpSNR_P  = 10*log10( var(tx(testIdx,1)) / var(z_hat(testIdx)-tx(testIdx,1)) );

% 3) BER (简单硬判决)
slice = @(r) sign(r).*(2*round((abs(r)-1)/2)+1);
berL = mean( slice(rxLegacy(testIdx,1)) ~= tx(testIdx,1) );
berP = mean( slice(z_hat(testIdx))      ~= tx(testIdx,1) );

%% ---------- 打印结果 ----------
fprintf('\n=== Simulation summary ===\n');
fprintf('FEXT suppression        : %.2f  dB\n', FEXTsupp_dB);
fprintf('DP-SNR  Legacy          : %.2f  dB\n', dpSNR_L);
fprintf('DP-SNR  Proposed        : %.2f  dB\n', dpSNR_P);
fprintf('BER     Legacy          : %.3e\n', berL);
fprintf('BER     Proposed        : %.3e\n', berP);

%% ---------- 绘图 ----------
figure('Name','Performance Comparison','Position',[300 200 900 450]);

subplot(1,3,1);
bar(FEXTsupp_dB); ylabel('dB'); title('FEXT suppression');
set(gca,'XTickLabel',{'Proposed'}); grid on;

subplot(1,3,2);
bar([dpSNR_L dpSNR_P]);
set(gca,'XTickLabel',{'Legacy','Proposed'});
ylabel('DP‑SNR [dB]'); title('DP‑SNR'); grid on;

subplot(1,3,3);
semilogy([berL berP],'o','MarkerSize',8); grid on;
set(gca,'XTick',1:2,'XTickLabel',{'Legacy','Proposed'});
ylabel('BER'); title('BER (log)');

sgtitle('Legacy vs. Proposed   |   10GBASE‑T Crosstalk Simulation');


%% =============== 新增部分：复杂度 & 延迟 =================
% ---- Legacy DSP tap 配置（若有不同实现，可自行改数字） ----
tapTHP_legacy   = 48;        % Tx 侧 THP FIR taps / pair
tapFFE_legacy   = 64;        % Rx 侧 FFE  taps / pair
tapDFE_legacy   = 128;       % Rx 侧 DFE  taps / pair

tapMISO_prop    = 12;        % Proposed：MISO FIR taps / pair  (示例)
% tapFFE (已在上方参数区给出)  = Proposed FFE taps / pair

% ---- 乘法器数量（四对并行时，线性相加）----
mulLegacy  = 4*(tapTHP_legacy + tapFFE_legacy + tapDFE_legacy);
mulPropose = 4*(tapMISO_prop  + tapFFE);

% ---- 算法级 Pipeline 延迟（tap 个数近似 UI）----
latLegacy  = (tapTHP_legacy + tapFFE_legacy + tapDFE_legacy);
latPropose = (tapMISO_prop  + tapFFE);

% ---- 打印 ----
fprintf('--- Complexity & Latency -----------------------------\n');
fprintf('Multipliers Legacy | Proposed : %d | %d  (%.0f %% fewer)\n',...
        mulLegacy, mulPropose, 100*(1-mulPropose/mulLegacy));
fprintf('Latency    Legacy | Proposed : %d | %d  UI  (%.0f %% shorter)\n',...
        latLegacy, latPropose, 100*(1-latPropose/latLegacy));

% ---- 画第二张图 ----
figure('Name','Complexity & Latency');

subplot(1,2,1);
bar([mulLegacy, mulPropose]);
set(gca,'XTickLabel',{'Legacy','Proposed'}); grid on;
ylabel('Real multipliers'); title('Arithmetic complexity');

subplot(1,2,2);
bar([latLegacy, latPropose]);
set(gca,'XTickLabel',{'Legacy','Proposed'}); grid on;
ylabel('Tap‑clock delay (UI)'); title('Algorithmic latency');

sgtitle('Legacy vs. Proposed  |  Complexity & Latency');


SNR_vec = 14:2:26;          % 7 个测试点
berL = zeros(size(SNR_vec));
berP = zeros(size(SNR_vec));

for s = 1:numel(SNR_vec)
    SNR_dB = SNR_vec(s);
    sigma  = 10^(-SNR_dB/20);
    ...    % 其余代码保持，但别把 figure 打开在 loop 中
    berL(s) = mean(slice(rxLegacy(testIdx,1))~=tx(testIdx,1));
    berP(s) = mean(slice(z_hat(testIdx))     ~=tx(testIdx,1));
end

figure; semilogy(SNR_vec, berL,'-o',SNR_vec, berP,'-s','LineWidth',1.3);
grid on; xlabel('Input SNR (dB)'); ylabel('BER');
legend({'Legacy','Proposed'},'Location','southwest');
title('BER vs. SNR  |  10GBASE‑T  (Cat‑6A 55 m, FEXT -25 dB)');

tapList = 4:4:32;                  % 8,12,16,...32
berTap  = zeros(size(tapList));  mulTap = zeros(size(tapList));

for t = 1:numel(tapList)
    tapMISO = tapList(t);
    ...     % 保持 SNR=20 dB，跑一遍
    berTap(t)  = mean(slice(z_hat(testIdx))~=tx(testIdx,1));
    mulTap(t)  = 4*(tapMISO+tapFFE);         % 乘法器计数
end

figure;
yyaxis left; semilogy(tapList, berTap,'-o'); ylabel('BER');
yyaxis right; plot(tapList, mulTap,'--s');   ylabel('Multipliers');
grid on; xlabel('MISO tap count');
title('Performance–Complexity trade‑off (Proposed chain)');
legend({'BER','Real mults'},'Location','east');
