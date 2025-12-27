% 10GBASE-T System Simulation with MISO Pre-compensation and FFE
% Date: March 11, 2025
% Description: Simulates a 10GBASE-T system integrating MISO pre-compensation
% and FFE equalization to cancel crosstalk and achieve a DP-SNR >= 23.4 dB.

clear; clc; close all;
rng(1); % 设置随机种子以确保可重复性

%% 1. 参数设置和比特生成
% LDPC 参数 (基于 10GBASE-T 的 LDPC(2048,1723))
n = 2048;              % 码字长度
k = 1723;              % 信息比特长度
bitsPerDSQ = 7;        % 每个 128-DSQ 符号携带 7 比特
numInfoBitsLDPC = 17230; % 示例：10 个码字的信息比特
numCodewords = ceil(numInfoBitsLDPC / k);
totalLDPCInfoBits = numCodewords * k; % 填充至完整的码字

% 采样和信道参数
fs = 800e6;            % 采样率：800 MHz
Ts = 1 / fs;           % 采样周期
EbN0_dB = 30;          % 信噪比（dB）

% 16-PAM 电平用于 128-DSQ 映射
pamLevels = -15:2:15;  % 16 个离散电平：[-15, -13, ..., 13, 15]

% 生成随机信息比特
txBitsLDPC = randi([0 1], numInfoBitsLDPC, 1);

% 填充比特以完成码字
padLength = totalLDPCInfoBits - numInfoBitsLDPC;
txBitsLDPC_padded = [txBitsLDPC; zeros(padLength, 1)];

%% 2. LDPC 编码
G_filename = 'C:\Users\Youan\OneDrive\桌面\Crosstalk\802.3an-2006\matrices\matrices\g.txt';
G = loadGMatrix(G_filename, k, n);
disp(['G 矩阵大小: ', num2str(size(G))]);
disp(['非零元素个数: ', num2str(nnz(G))]);
codedBits = [];
for i = 1:numCodewords
    infoBlock = txBitsLDPC_padded((i-1)*k+1 : i*k);
    codeword = mod(infoBlock' * G, 2)';
    codedBits = [codedBits; codeword];
end

%% 3. 128-DSQ 调制
txSymbols = dsq128Mapper(codedBits, bitsPerDSQ, pamLevels);
% txSymbols 是 N x 2 矩阵，每行是一个 DSQ 符号对

% 形成复基带信号
txSignal = txSymbols(:,1) + 1j * txSymbols(:,2);

% 测试调制和解调
testBits = codedBits(1:700); % 取前 100 个符号的比特 (100 * 7 = 700)
testSymbols = dsq128Mapper(testBits, bitsPerDSQ, pamLevels);
testRecoveredBits = dsq128Demapper(testSymbols, pamLevels);
testBER = biterr(testBits, testRecoveredBits) / length(testBits);
fprintf('调制-解调测试 BER: %.2e\n', testBER);

%% 4. 信道建模和 MISO 预补偿
% 定义一个简化的 4x4 MIMO 信道，使用 FIR 滤波器
channelOrder = 50;
% 信道 1 的冲激响应
h11 = fir2(channelOrder, [0 1], [1 0.1]); h11 = h11 / norm(h11); % 主信道
h12 = 0.2 * fir2(channelOrder, [0 1], [1 0.05]); h12 = h12 / norm(h12); % 来自信道 2 的远端串扰
h13 = 0.15 * fir2(channelOrder, [0 1], [1 0.03]); h13 = h13 / norm(h13); % 来自信道 3 的远端串扰
h14 = 0.1 * fir2(channelOrder, [0 1], [1 0.02]); h14 = h14 / norm(h14); % 来自信道 4 的远端串扰

% 模拟其他信道的发送信号（占位符）
txSignal2 = txSignal; % 信道 2 信号
txSignal3 = txSignal; % 信道 3 信号
txSignal4 = txSignal; % 信道 4 信号

% 使用零强制法进行 MISO 预补偿
% 在频域计算预补偿滤波器 l(1,j) = H1j / H11
fftLen = 1024;
freqResp_h11 = fft(h11, fftLen);
freqResp_h12 = fft(h12, fftLen);
freqResp_h13 = fft(h13, fftLen);
freqResp_h14 = fft(h14, fftLen);

% 添加正则化项以避免除以小值
epsilon = 1e-6;
l12_freq = freqResp_h12 ./ (freqResp_h11 + epsilon);
l13_freq = freqResp_h13 ./ (freqResp_h11 + epsilon);
l14_freq = freqResp_h14 ./ (freqResp_h11 + epsilon);

l12 = real(ifft(l12_freq, fftLen));
l13 = real(ifft(l13_freq, fftLen));
l14 = real(ifft(l14_freq, fftLen));

% 截断滤波器
filterLength = 20;
l12 = l12(1:filterLength);
l13 = l13(1:filterLength);
l14 = l14(1:filterLength);

% 对信道 1 应用预补偿
u1 = txSignal - filter(l12, 1, txSignal2) - filter(l13, 1, txSignal3) - filter(l14, 1, txSignal4);

% 信道 1 的接收信号
y1 = filter(h11, 1, u1) + filter(h12, 1, txSignal2) + filter(h13, 1, txSignal3) + filter(h14, 1, txSignal4);

% 验证预补偿效果
y1_ideal = filter(h11, 1, txSignal); % 理想接收信号（无串扰）
y1_no_noise = filter(h11, 1, u1) + filter(h12, 1, txSignal2) + filter(h13, 1, txSignal3) + filter(h14, 1, txSignal4);
mse_precomp = mean(abs(y1_no_noise - y1_ideal).^2);
fprintf('预补偿后 MSE: %.2e\n', mse_precomp);

% 添加 AWGN
signalPower = var(y1);
noisePower = signalPower / (10^(EbN0_dB / 10));
thermalNoise = sqrt(noisePower) * (randn(length(y1), 1) + 1j * randn(length(y1), 1)) / sqrt(2);
y1 = y1 + thermalNoise;

%% 5. 接收端 FFE 均衡
FFE_order = 20; % 增加 FFE 阶数
L_f = FFE_order + 1;
H_conv = convmtx(h11, L_f); % 主信道的卷积矩阵

% 期望响应（延迟的 delta 函数）
[~, peak_idx] = max(abs(h11));
delay = peak_idx + floor(FFE_order / 2);
if delay > size(H_conv, 1)
    delay = size(H_conv, 1);
end
d_vec = zeros(size(H_conv, 1), 1);
d_vec(delay) = 1;

% 零强制 FFE 系数
f = pinv(H_conv) * d_vec;

% 应用 FFE
ffeOutput = filter(f, 1, y1);

%% 6. DSQ 解调和比特恢复
numSymbols = floor(length(ffeOutput) / 2);
ffeOutputPairs = reshape(ffeOutput(1:2*numSymbols), 2, numSymbols).';

% 解调至比特
rxBits = dsq128Demapper(ffeOutputPairs, pamLevels);

% 截断至原始长度
minLen = min(length(rxBits), numInfoBitsLDPC);
rxDecodedBits = rxBits(1:minLen);

%% 7. 性能评估
% 误比特率 (BER)
[errCount, ber] = biterr(rxDecodedBits, txBitsLDPC(1:minLen));
fprintf('BER: %.2e (错误数: %d, 比特数: %d)\n', ber, errCount, minLen);

% 判决点信噪比 (DP-SNR)
% 补偿 FFE 引入的延迟
ffeDelay = delay - 1; % FFE 滤波器的延迟
if ffeDelay > 0
    txSignalAligned = txSignal(ffeDelay+1:end);
    ffeOutputTrimmed = ffeOutput(1:length(txSignalAligned));
else
    txSignalAligned = txSignal(1:length(ffeOutput));
    ffeOutputTrimmed = ffeOutput;
end

% 计算 DP-SNR
errorSignal = ffeOutputTrimmed - txSignalAligned;
signalPower = mean(abs(txSignalAligned).^2);
noisePower = mean(abs(errorSignal).^2);
dp_snr = 10 * log10(signalPower / noisePower);
fprintf('DP-SNR: %.2f dB\n', dp_snr);

%% 8. 可视化
figure;
scatter(real(ffeOutputPairs(:,1)), real(ffeOutputPairs(:,2)), '.', 'DisplayName', '接收信号');
hold on;
scatter(txSymbols(:,1), txSymbols(:,2), 'ro', 'filled', 'DisplayName', '理想信号');
title('接收信号 vs 理想 128-DSQ 星座图');
xlabel('PAM16 维度 1');
ylabel('PAM16 维度 2');
legend;
grid on;

disp('仿真完成！');

%% 函数定义
function G = loadGMatrix(filename, numRows, numCols)
    % 打开文件
    fid = fopen(filename, 'r');
    if fid == -1
        error('无法打开文件: %s', filename);
    end
    
    % 读取所有行
    G_lines = textscan(fid, '%s', 'Delimiter', '\n');
    fclose(fid);
    G_lines = G_lines{1};
    
    % 初始化稀疏矩阵的行和列索引
    row_indices = [];
    col_indices = [];
    
    % 解析每行
    for i = 1:numRows
        line = strtrim(G_lines{i});
        if ~isempty(line)
            % 解析列索引（基于 0 的索引）
            indices = str2num(line); %#ok<ST2NM>
            % 转换为基于 1 的索引
            col_indices = [col_indices, indices + 1]; %#ok<AGROW>
            row_indices = [row_indices, i * ones(1, length(indices))]; %#ok<AGROW>
        end
    end
    
    % 构建稀疏矩阵
    G = sparse(row_indices, col_indices, 1, numRows, numCols);
    
    % 验证矩阵
    fprintf('G 矩阵维度: %d×%d\n', size(G, 1), size(G, 2));
    fprintf('非零元素数量: %d\n', nnz(G));
    if nnz(G) == 0
        warning('G 矩阵加载为空，请检查文件内容或路径: %s', filename);
    end
end

function mappedSymbols = dsq128Mapper(codedBits, bitsPerDSQ, pamLevels)
    % 确保 pamLevels 是行向量
    pamLevels = pamLevels(:)';
    
    % 计算符号数量
    numSymbols = floor(length(codedBits) / bitsPerDSQ);
    
    % 截断多余比特以对齐符号边界
    codedBits = codedBits(1:numSymbols * bitsPerDSQ);
    
    % 将比特重塑为 numSymbols x bitsPerDSQ 矩阵
    bitMatrix = reshape(codedBits, bitsPerDSQ, numSymbols)';
    
    % 初始化输出符号矩阵
    mappedSymbols = zeros(numSymbols, 2);
    
    % 格雷码映射表 (4 比特到 16-PAM)
    grayMap = [0 1 3 2 6 7 5 4 12 13 15 14 10 11 9 8]; % 格雷码顺序
    pamIndices = 0:15;
    pamMapping = zeros(1, 16);
    for i = 0:15
        pamMapping(grayMap(i+1)+1) = pamIndices(i+1);
    end
    
    % 将比特映射到符号
    for i = 1:numSymbols
        bits = bitMatrix(i, :);
        
        % 将前 4 比特映射到 I 分量 (16-PAM)
        I_bits = bits(1:4);
        I_val = bi2de(I_bits, 'left-msb'); % 转换为十进制 (0 到 15)
        I_val = pamMapping(I_val + 1); % 应用格雷码映射
        mappedSymbols(i, 1) = pamLevels(I_val + 1); % 映射到 PAM 电平
        
        % 将后 3 比特和第 1 比特映射到 Q 分量 (16-PAM)
        Q_bits = [bits(1), bits(5:7)];
        Q_val = bi2de(Q_bits, 'left-msb'); % 转换为十进制 (0 到 15)
        Q_val = pamMapping(Q_val + 1); % 应用格雷码映射
        mappedSymbols(i, 2) = pamLevels(Q_val + 1); % 映射到 PAM 电平
    end
end

function rxBits = dsq128Demapper(symbols, pamLevels)
    % 符号数量
    numSymbols = size(symbols, 1);
    bitsPerDSQ = 7;
    rxBits = zeros(numSymbols * bitsPerDSQ, 1);
    
    % 格雷码映射表 (4 比特到 16-PAM)
    grayMap = [0 1 3 2 6 7 5 4 12 13 15 14 10 11 9 8]; % 格雷码顺序
    invGrayMap = zeros(1, 16);
    for i = 0:15
        invGrayMap(grayMap(i+1)+1) = i;
    end
    
    % 处理每个符号对
    for i = 1:numSymbols
        % 为 I 和 Q 找到最近的 PAM 电平索引
        [~, I_idx] = min(abs(pamLevels - symbols(i, 1)));
        [~, Q_idx] = min(abs(pamLevels - symbols(i, 2)));
        
        % 转换为基于 0 的索引
        I_val = I_idx - 1; % 0 到 15
        Q_val = Q_idx - 1; % 0 到 15
        
        % 反向格雷码映射
        I_val = invGrayMap(I_val + 1);
        Q_val = invGrayMap(Q_val + 1);
        
        % 将索引转换为 4 比特二进制
        I_bits = de2bi(I_val, 4, 'left-msb');
        Q_bits = de2bi(Q_val, 4, 'left-msb');
        
        % 重构 7 比特序列
        bits = [I_bits(1); I_bits(2:4)'; Q_bits(2:4)'];
        
        % 存储比特
        rxBits((i-1)*bitsPerDSQ + 1 : i*bitsPerDSQ) = bits;
    end
end