%%  ======  MISO_LMS.m  ======
% 10GBASE-T  MISO 预补偿 + FFE (LMS) 串扰消除仿真
% 修订日期: 2025-04-24
% 作者: Zheng Youan  (郑佑安)

clear; clc; close all;
rng(1);                         % 可重复随机序列
fprintf('--- MISO+FFE Crosstalk Simulation  (10GBASE-T) ---\n');

%% 1. 基本参数 ===========================================================
k  = 1723;        % 信息比特
n  = 2048;        % 码字长度
bitsPerDSQ = 7;   % 128-DSQ 每符号 7 比特
pamLevels   = -15:2:15;         % PAM-16 电平
fs          = 800e6;            % 800 MHz 采样率
Ts          = 1/fs;
EbN0_dB     = 30;               % Eb/N0
N_train     = 1e4;              % 训练符号数
L           = 10;               % MISO 抽头
M           = 6;                % FFE 抽头
mu_MISO     = 1e-3;             % LMS 步长（MISO）
mu_FFE      = 1e-2;             % LMS 步长（FFE）
numCodewords_data = 10;         % 数据阶段码字数
numInfoBits_data  = k * numCodewords_data;

%% 2. 载入 (2048,1723) LDPC 生成矩阵 G ================================
try
    % 若同目录存在 g.txt 直接读取，否则弹窗手选
    defaultFile = fullfile(pwd,'g.txt');
    if ~isfile(defaultFile)
        [f,p] = uigetfile('*.txt','请选择 10GBASE-T g.txt');
        defaultFile = fullfile(p,f);
    end
    G = loadGMatrix(defaultFile,k,n);               % ← 修订后的函数
catch ME
    error('G 矩阵加载失败: %s',ME.message);
end
fprintf('G 维度: %d × %d, 稀疏度: %.4f\n',size(G,1),size(G,2),nnz(G)/(numel(G)));

%% 3. 双通道 (主通道 + 串扰) 建模  ======================================
chanOrd = 10;
h11 = fir2(chanOrd,[0 1],[1 0.1]);  h11 = h11.'/norm(h11);     % 列向量
h12 = 0.5*fir2(chanOrd,[0 1],[1 0.05]); h12 = h12.'/norm(h12); % 列向量

%% 4. 噪声功率
sigPow = mean(abs(filter(h11,1,randn(1e4,1)+1j*randn(1e4,1))).^2);
noisePow = sigPow / (10^(EbN0_dB/10));

%% 5. ------ 训练阶段 (联合自适应) --------------------------------------
% (1) 训练比特生成并 LDPC 编码
Nbits_train = ceil(N_train / floor(n/bitsPerDSQ))*k;
txBits_train1 = randi([0 1],Nbits_train,1);
txBits_train2 = randi([0 1],Nbits_train,1);

codedBits_train1 = encodeBlocks(txBits_train1,G,k,n);
codedBits_train2 = encodeBlocks(txBits_train2,G,k,n);

% (2) 128-DSQ 映射
txSym_train1 = dsq128Mapper(codedBits_train1,bitsPerDSQ,pamLevels);
txSym_train2 = dsq128Mapper(codedBits_train2,bitsPerDSQ,pamLevels);
s1 = txSym_train1(:,1)+1j*txSym_train1(:,2);   s1 = s1(1:N_train);
s2 = txSym_train2(:,1)+1j*txSym_train2(:,2);   s2 = s2(1:N_train);

% (3) LMS 训练
l12 = zeros(L,1);          % MISO 预补偿滤波器 (列向量)
w_miso = zeros(M,1);       % FFE
state11 = zeros(length(h11)-1,1);
state12 = zeros(length(h12)-1,1);
yTrain  = zeros(N_train,1);

% 计算 FFE 延迟 (峰值 + (M-1)/2)
[~,pidx] = max(abs(h11));  D = pidx+floor((M-1)/2)-1;

for ksym = 1:N_train
    % ---------- MISO 预补偿 ----------
    x2_vec = flipud(padarray(s2(max(1,ksym-L+1):ksym),L - min(L,ksym),0,'pre'));
    u1 = s1(ksym) - (l12.' * x2_vec);                 % 标量
    
    [yMain,state11] = filter(h11,1,u1,state11);       % 主通道
    [yXT ,state12]  = filter(h12,1,s2(ksym),state12); % 串扰
    y = yMain + yXT + sqrt(noisePow/2)*(randn+1j*randn);
    yTrain(ksym) = y;
    
    % ---------- FFE ----------
    y_vec = flipud(padarray(yTrain(max(1,ksym-M+1):ksym),M - min(M,ksym),0,'pre'));
    y_hat = w_miso.' * y_vec;
    
    if ksym > D
        d = s1(ksym-D);                % 期望符号
        e = d - y_hat;                 % 误差
        w_miso = w_miso + mu_FFE*conj(e)*y_vec;
        l12    = l12  - mu_MISO*conj(e)*x2_vec;
    end
end
fprintf('训练完成: 自适应滤波器已收敛。\n');

%% 6. ------ 数据阶段 ---------------------------------------------------
txBits_data  = randi([0 1],numInfoBits_data,1);
txBits_data2 = randi([0 1],numInfoBits_data,1);

codedBits_data  = encodeBlocks(txBits_data ,G,k,n);
codedBits_data2 = encodeBlocks(txBits_data2,G,k,n);

txSym_data  = dsq128Mapper(codedBits_data ,bitsPerDSQ,pamLevels);
txSym_data2 = dsq128Mapper(codedBits_data2,bitsPerDSQ,pamLevels);

s1d = txSym_data (:,1)+1j*txSym_data (:,2);
s2d = txSym_data2(:,1)+1j*txSym_data2(:,2);

% ----------- 预补偿并通过信道 ----------
u1d = s1d - filter(l12,1,s2d);                       % 线性预补偿
y_d = filter(h11,1,u1d) + filter(h12,1,s2d);
y_d = y_d + sqrt(noisePow/2)*(randn(size(y_d))+1j*randn(size(y_d)));
y_out = filter(w_miso,1,y_d);

%% 7. 解映射 / BER / DP-SNR
symPairs = reshape(y_out(1:2*floor(length(y_out)/2)),2,[]).';
rxBits   = dsq128Demapper(symPairs,pamLevels);
% rxBits   = rxBits(1:numInfoBits_data);

% ---- 解映射后长度对齐 -----------------------------------------------
minLen = min(length(rxBits), numInfoBits_data);   % 取公共最短
rxBits_cmp = rxBits       (1:minLen);
txBits_cmp = txBits_data  (1:minLen);

% ---- 误比特率 --------------------------------------------------------
[errCnt, ber] = biterr(rxBits_cmp, txBits_cmp);
fprintf('MISO+FFE  数据 BER = %.2e  (错误 %d / %d)\n', ...
        ber, errCnt, minLen);

% DP-SNR
if D>0, ref = s1d(D+1:end);  ycmp = y_out(1:length(ref));
else,   ref = s1d;           ycmp = y_out(1:length(ref));
end
dp_snr = 10*log10(mean(abs(ref).^2) / mean(abs(ycmp-ref).^2));
% %%%%%fprintf('DP-SNR = %.2f dB  (目标 ≥23.4 dB)\n',dp_snr);

%% 8. 可视化 (星座)
figure;  scatter(real(symPairs(:,1)),real(symPairs(:,2)),'.'); hold on;
scatter(txSym_data(:,1),txSym_data(:,2),'ro'); grid on;
title('接收-理想 128-DSQ 星座对比'); legend('接收符号','理想符号');

disp('仿真结束。');

%% ======================================================================
%                 ——  函  数  定  义  ——                              %%
% =======================================================================

function G = loadGMatrix(fname,k,n)
% 读取 10GBASE-T generator matrix (g.txt, 每行为空格分隔列索引 0-based)
fid = fopen(fname,'r');  assert(fid~=-1,'无法打开 %s',fname);
C  = textscan(fid,'%s','Delimiter','\n'); fclose(fid);
raw = C{1};   raw(cellfun(@isempty,raw)) = [];        % 去空行
assert(numel(raw)>=k,'文件行数不足: 期望 %d 行',k);

row_idx = []; col_idx = [];
for r = 1:k
    cols = sscanf(raw{r},'%d').'+1;                  % 0-based → 1-based
    row_idx = [row_idx, r*ones(1,numel(cols))];      %#ok<AGROW>
    col_idx = [col_idx, cols                 ];      %#ok<AGROW>
end
G = sparse(row_idx,col_idx,1,k,n);

% 若维度反了自动转置
if ~(size(G,1)==k && size(G,2)==n)
    warning('G 维度 %d×%d 与期望不符，自动转置……',size(G));
    G = G.';
    assert(size(G,1)==k && size(G,2)==n,...
        '转置后维度仍不匹配，请检查 g.txt 与 k,n 设置');
end
end

function cw = encodeLDPC(infoBits,G,k,n)
% 支持 G 为 k×n 或 n×k
if size(G,1)~=k || size(G,2)~=n
    if size(G,1)==n && size(G,2)==k
        G = G.';                                     % 转置修正
    else
        error('G 维度与 (k,n) 不匹配');
    end
end
cw = mod(infoBits.'*G,2).';                          % 列向量
end

function outBits = encodeBlocks(inBits,G,k,n)
numBlk = length(inBits)/k;  assert(rem(numBlk,1)==0,'输入长度必须是 k 的整数倍');
outBits = zeros(numBlk*n,1);
for b = 1:numBlk
    seg = inBits((b-1)*k+1:b*k);
    outBits((b-1)*n+1:b*n) = encodeLDPC(seg,G,k,n);
end
end

function symOut = dsq128Mapper(bits,bitsPerDSQ,levels)
levels = levels(:).';
numSym = floor(length(bits)/bitsPerDSQ);
bits   = reshape(bits(1:numSym*bitsPerDSQ),bitsPerDSQ,numSym).';
symOut = zeros(numSym,2);

gray = bitxor((0:15),floor((0:15)/2));       % Gray 映射表
invG = zeros(1,16);  invG(gray+1) = 0:15;

for i = 1:numSym
    b = bits(i,:);
    I = invG(bi2de(b(1:4), 'left-msb')+1);
    Q = invG(bi2de([b(1) b(5:7)],'left-msb')+1);
    symOut(i,1) = levels(I+1);
    symOut(i,2) = levels(Q+1);
end
end

function rxBits = dsq128Demapper(sym,levels)
numSym = size(sym,1);
rxBits  = zeros(numSym*7,1);
gray = bitxor((0:15),floor((0:15)/2)); invG = zeros(1,16); invG(gray+1)=0:15;
for i = 1:numSym
    [~,Iidx] = min(abs(levels-real(sym(i,1)))); I = gray(Iidx-1+1);
    [~,Qidx] = min(abs(levels-real(sym(i,2)))); Q = gray(Qidx-1+1);
    Ibits = de2bi(I,4,'left-msb');
    Qbits = de2bi(Q,4,'left-msb');
    rxBits((i-1)*7+1:i*7) = [Ibits(1) Ibits(2:4) Qbits(2:4)].';
end
end
