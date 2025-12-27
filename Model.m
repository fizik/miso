% 10GBASE-T System Simulation Script
% Date: March 11, 2025
% Function: Simulate a 10GBASE-T system (per IEEE 802.3an-2006) integrating
% LDPC encoding (via generator matrix G from G.txt), 128-DSQ modulation,
% channel modeling, receiver equalization (FFE+DFE), DSQ demapping, and performance evaluation.
% Each major section includes visualization for understanding.

clear; clc; close all;
rng(1); % 固定随机种子

%% Part 1: Parameter Settings and Bit Generation
% LDPC parameters (LDPC(2048,1723))
n = 2048;              % 码字长度
k = 1723;              % 信息比特长度
% DSQ modulation: 每个 DSQ 符号携带 7 比特
bitsPerDSQ = 7;
% 假设总LDPC信息比特数
numInfoBitsLDPC = 17230; % 比如10个码字
numCodewords = ceil(numInfoBitsLDPC/k);
totalLDPCInfoBits = numCodewords * k;  % 补齐后的信息比特数

% 采样及信道参数
fs = 800e6;            % 采样率 800 MHz
Ts = 1/fs;
EbN0_dB = 30;          % 信噪比(dB)

% 16-PAM 电平（用于 DSQ 映射）
pamLevels = -15:2:15;  % 16 个离散电平

% 生成随机信息比特 (Part 1 Visualization: 比特分布直方图)
txBitsLDPC = randi([0 1], numInfoBitsLDPC, 1);
figure;
histogram(txBitsLDPC, 'Normalization', 'probability');
title('LDPC Information Bit Distribution');
xlabel('Bit Value'); ylabel('Probability');

% 填充至完整码字
padLength = totalLDPCInfoBits - numInfoBitsLDPC;
txBitsLDPC_padded = [txBitsLDPC; zeros(padLength, 1)];

%% Part 2: LDPC Encoding Using Generator Matrix G
% 本例使用 G.txt 文件（生成器矩阵），假设 G.txt 有 1723 行，每行给出该行中“1”条目的列索引（0-based）
% 请修改路径为您的实际文件路径：
G_filename = 'C:\Users\Youan\OneDrive\桌面\Crosstalk\802.3an-2006\matrices\matrices\G.txt';
G = loadGMatrix(G_filename, k, n); % 得到一个 k x n 的稀疏二值矩阵

% 对每个信息块进行编码：码字 = info * G (模2)
codedBits = [];
for i = 1:numCodewords
    infoBlock = txBitsLDPC_padded((i-1)*k+1 : i*k);
    % 使用矩阵乘法进行编码（模2）
    codeword = mod(infoBlock' * G, 2)';
    codedBits = [codedBits; codeword];
end

% Part 2 Visualization: 显示部分码字（例如前50比特）
disp('前50个码字比特:');
disp(codedBits(1:50));

%% Part 3: 128-DSQ Modulation Mapping
% 将编码比特映射为 128-DSQ 符号，每 7 比特映射为一个 DSQ 符号（由两个 16-PAM 符号构成）
totalBitsTx = codedBits; % 此处仅对编码比特调制
txSymbols = dsq128Mapper(totalBitsTx, bitsPerDSQ, pamLevels);
% 可视化 DSQ 映射结果：散点图展示映射的二维星座点
figure;
scatter(txSymbols(:,1), txSymbols(:,2), 'filled');
title('Ideal 128-DSQ Constellation (Mapped Symbols)');
xlabel('PAM16 Dimension 1'); ylabel('PAM16 Dimension 2');

% 构造复数基带信号：实部为第一个 PAM 符号，虚部为第二个
txSignal = txSymbols(:,1) + 1j*txSymbols(:,2);

%% Part 4: Channel Modeling
% 构造主信道：使用 FIR 滤波器模拟插入损耗和 ISI
channelOrder = 50;
channelImpulse = fir2(channelOrder, [0 1], [1 0.1]);
channelImpulse = channelImpulse / norm(channelImpulse);
mainSignal = conv(txSignal, channelImpulse, 'same');

% 可视化主信道冲激响应
figure;
stem(channelImpulse);
title('Channel Impulse Response');
xlabel('Sample Index'); ylabel('Amplitude');

% 构造 NEXT 串扰（近端串扰）：短延时，低幅值
nextCoupling = [0.01 -0.005 0.002];
nextDelay = 1;
nextInterference = filter(nextCoupling, 1, txSignal);
nextInterference = [zeros(nextDelay,1); nextInterference(1:end-nextDelay)];
nextInterference = 0.1 * nextInterference;

% 构造 FEXT 串扰（远端串扰）：较长延时，适中幅值
fextCoupling = channelImpulse * 0.2;
fextDelay = 100;
fextInterference = filter(fextCoupling, 1, txSignal);
fextInterference = [zeros(fextDelay,1); fextInterference(1:end-fextDelay)];
fextInterference = 0.2 * fextInterference;

% 外部串扰（Alien NEXT）：额外高斯噪声
alienNoise = 0.01 * randn(length(txSignal),1);

% AWGN：根据目标 Eb/N0 添加热噪声
signalPower = var(mainSignal);
noisePower = signalPower / (10^(EbN0_dB/10));
thermalNoise = sqrt(noisePower) * randn(length(txSignal),1);

% 组合各信号，得到接收信号
rxSignal = mainSignal + nextInterference + fextInterference + alienNoise + thermalNoise;

% Part 4 Visualization: 绘制发射信号与接收信号的幅度曲线（前 200 个采样点）
figure;
plot(abs(txSignal(1:200)), 'b.-'); hold on;
plot(abs(rxSignal(1:200)), 'r.-');
legend('Transmitted Signal','Received Signal');
title('Signal Amplitude (First 200 Samples)');
xlabel('Sample Index'); ylabel('Amplitude');

%% Part 5: Receiver Equalization (FFE + DFE)
% 设计 FFE 使用 convmtx 构造卷积矩阵求解零强迫（ZF）系数
FFE_order = 5;         % FFE 阶数
L_f = FFE_order + 1;   % FFE 滤波器长度
L = length(channelImpulse);  % 信道冲激响应长度

% 确保 channelImpulse 为列向量
channelImpulse = channelImpulse(:);
H_conv = convmtx(channelImpulse, L_f);  % (L+L_f-1) x L_f 矩阵，应为 56 x 6

% 调试：检查尺寸
disp(['H_conv size after correction: ', num2str(size(H_conv))]);
disp(['Expected H_conv size: ', num2str([L + L_f - 1, L_f])]);

% 选择期望延时 d0
d0 = L_f; % 选择延迟为 L_f
d_vec = zeros(L + L_f - 1, 1);
d_vec(d0) = 1;
disp(['d_vec size: ', num2str(size(d_vec))]);

% 求解 FFE 系数
try
    fCoeffs = H_conv \ d_vec;
catch
    warning('直接求解失败，尝试使用伪逆！');
    fCoeffs = pinv(H_conv) * d_vec;
end
f = fCoeffs(:).';  % FFE 系数，长度 L_f

% 可视化 FFE 系数
figure;
stem(f);
title('FFE Filter Coefficients');
xlabel('Tap Index'); ylabel('Coefficient');

% 简单 DFE 系数设计：取信道冲激响应主峰后 DFE_order 个抽头
DFE_order = 3; 
if L < (DFE_order+1)
    error('信道冲激响应长度不足以设计 DFE 系数');
end
postCursor = channelImpulse(2:DFE_order+1);
b = postCursor.';  % DFE 系数

% 对接收信号应用 FFE 滤波
ffeOutput = filter(f, 1, rxSignal);

% DFE 判决：逐个符号反馈决策
numSymbols = length(ffeOutput);
decisions = zeros(numSymbols,1);
for n = DFE_order+1:numSymbols
    dfe_out = ffeOutput(n);
    for k = 1:DFE_order
        if n-k >= 1
            dfe_out = dfe_out - b(k) * decisions(n-k);
        end
    end
    % 近邻判决
    [~, idx] = min(abs(pamLevels - real(dfe_out)));
    decisions(n) = pamLevels(idx);
end

% 如果决策数为奇数，则裁剪最后一元素以便重组为符号对，并同步裁剪ffeOutput和numSymbols
if mod(length(decisions),2) ~= 0
    decisions = decisions(1:end-1);
    ffeOutput = ffeOutput(1:end-1); % 同步裁剪FFE输出
    numSymbols = length(ffeOutput); % 同步更新numSymbols
end
decisionsPairs = reshape(decisions, 2, []).';

% 调试：验证长度一致性
disp(['numSymbols after update: ', num2str(numSymbols)]);
disp(['length(decisions) after update: ', num2str(length(decisions))]);
disp(['length(ffeOutput) after update: ', num2str(length(ffeOutput))]);

% Part 5 Visualization:
% 绘制 FFE 输出信号的前 200 个样本以及均衡后判决
figure;
subplot(2,1,1);
plot(real(ffeOutput(1:200)), 'b.-');
title('FFE Output (First 200 Samples)');
xlabel('Sample Index'); ylabel('Amplitude');

subplot(2,1,2);
plot(decisions(1:200), 'r.-');
title('Decisions (First 200 Samples)');
xlabel('Sample Index'); ylabel('Decision Value');

%% Part 6: DSQ Demapping and Bit Recovery
rxBits = dsq128Demapper(decisionsPairs, pamLevels);

% 注意：由于调制映射是按每 7 比特进行，解调恢复的比特数可能为 floor(totalBitsTx/7)*7
recoveredBits = rxBits;  
% 若恢复比特数少于原始信息比特数，则取最小长度比较
minLen = min(length(recoveredBits), numInfoBitsLDPC);
rxDecodedBits = recoveredBits(1:minLen);

% Part 6 Visualization: 显示前200个恢复的比特（以 0 和 1 展示）
figure;
plot(rxDecodedBits(1:200), 'b.-');
title('Recovered Bits (First 200 bits)');
xlabel('Bit Index'); ylabel('Bit Value');

%% Part 7: Performance Evaluation
% 计算 BER
minLen2 = min(length(rxDecodedBits), length(txBitsLDPC));
[errCount, ber] = biterr(rxDecodedBits(1:minLen2), txBitsLDPC(1:minLen2));
fprintf('LDPC BER: %.2e (错误数 = %d, 比特数 = %d)\n', ber, errCount, minLen2);

% 计算决策点信噪比 (DP-SNR)
errorSignal = zeros(numSymbols,1);
for n = 1:numSymbols
    dfe_out = ffeOutput(n);
    for k = 1:DFE_order
        if n-k >= 1
            dfe_out = dfe_out - b(k) * decisions(n-k);
        end
    end
    errorSignal(n) = real(dfe_out) - decisions(n); % 取实部
end
signal_power = mean(decisions.^2);
noise_power = mean(errorSignal.^2);
dp_snr = 10*log10(signal_power/noise_power);
fprintf('Decision Point SNR: %.2f dB\n', dp_snr);

% Part 7 Visualization: 显示误差信号直方图
figure;
histogram(errorSignal, 50);
title('Histogram of Equalization Error Signal');
xlabel('Error Value'); ylabel('Frequency');

%% Part 8: Signal Analysis (PSD and Constellation Diagram)
% 8.1 PSD of Transmit Signal
% 调试：检查信号长度
disp(['length(txSignal): ', num2str(length(txSignal))]);
windowLength = 512; % 调整窗口长度小于信号长度
[PSD_tx, faxis] = pwelch(txSignal, windowLength, [], windowLength, fs, 'twosided');
figure;
plot(faxis/1e6, 10*log10(PSD_tx));
title('Transmit Signal PSD');
xlabel('Frequency (MHz)');
ylabel('PSD (dB/Hz)');

% 8.2 Received Constellation Diagram from FFE Output
% 将 FFE 输出重组成 DSQ 符号对（用于星座图绘制）
reshapedFFE = reshape(ffeOutput, 2, []).';
figure;
scatter(real(reshapedFFE(:,1)), real(reshapedFFE(:,2)), '.');
hold on;
scatter(txSymbols(:,1), txSymbols(:,2), 'ro', 'filled');
title('Received Constellation (FFE Output)');
xlabel('PAM16 Dimension 1');
ylabel('PAM16 Dimension 2');
legend('Received Symbols','Ideal DSQ128 Constellation');

disp('Simulation completed!');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Function Definitions

function G = loadGMatrix(filename, numRows, numCols)
    % Load generator matrix G from G.txt file.
    % Each line in the file corresponds to one row of G.
    % Each line contains integers (space-separated) representing the
    % column indices (0-based) of 1's in that row.
    fid = fopen(filename, 'r');
    if fid == -1
        error('无法打开文件: %s', filename);
    end
    G_data = textscan(fid, '%s', 'Delimiter', '\n');
    fclose(fid);
    G_lines = G_data{1};
    G = sparse(numRows, numCols);
    for i = 1:numRows
        line = strtrim(G_lines{i});
        if isempty(line)
            continue;
        end
        indices = str2num(line); %#ok<ST2NM>
        if ~isempty(indices)
            G(i, indices+1) = 1; % 转换为 MATLAB 的 1-based 索引
        end
    end
end

function mappedSymbols = dsq128Mapper(codedBits, bitsPerDSQ, pamLevels)
    % Map input bits into 128-DSQ symbols.
    % 每个 DSQ 符号由 bitsPerDSQ (7) 比特构成：
    % 第1比特为 LSB，要求两个 PAM 符号共享；第2-4比特用于第一个 PAM，5-7用于第二个 PAM。
    numSymbols = floor(length(codedBits)/bitsPerDSQ);
    mappedSymbols = zeros(numSymbols, 2);
    bitIndex = 1;
    for i = 1:numSymbols
        b = codedBits(bitIndex:bitIndex+bitsPerDSQ-1);
        bitIndex = bitIndex + bitsPerDSQ;
        % 简单映射：将前4比特和后4比特（第1比特重复）转化为 0~15 的整数
        first_val = b(4)*8 + b(3)*4 + b(2)*2 + b(1);
        second_val = b(7)*8 + b(6)*4 + b(5)*2 + b(1);
        mappedSymbols(i,:) = [pamLevels(first_val+1), pamLevels(second_val+1)];
    end
end

function rxBits = dsq128Demapper(symbols, pamLevels)
    % Demap DSQ symbols back into bits.
    % 每行 symbols 为 [I, Q]，逆映射回 7 比特。
    numSymbols = size(symbols,1);
    rxBits = [];
    for i = 1:numSymbols
        x_val = symbols(i,1);
        y_val = symbols(i,2);
        % 将 PAM 电平反映射到 0~15 的索引
        [~, x_idx] = min(abs(pamLevels - x_val));
        [~, y_idx] = min(abs(pamLevels - y_val));
        x_idx = x_idx - 1;
        y_idx = y_idx - 1;
        lsb = mod(x_idx, 2); % 假定两个符号的LSB一致
        x_high = floor(x_idx/2);
        y_high = floor(y_idx/2);
        bits_x = de2bi(x_high, 3, 'left-msb')';
        bits_y = de2bi(y_high, 3, 'left-msb')';
        rxSymbolBits = [lsb; bits_x; bits_y]; % 7比特
        rxBits = [rxBits; rxSymbolBits];
    end
end