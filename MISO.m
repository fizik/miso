% Simplified 10GBASE-T System Simulation to Compare MISO vs Standard Model
% Date: April 01, 2025
% Description: Simulates a simplified 10GBASE-T system to compare MISO pre-compensation
% with a standard model (no pre-compensation) in terms of BER, DP-SNR, and visualization.

clear; clc; close all;
rng(1); % Set random seed for reproducibility

%% 1. Parameter Settings and Bit Generation
% LDPC parameters (based on 10GBASE-T LDPC(2048,1723))
n = 2048;              % Codeword length
k = 1723;              % Information bit length
bitsPerDSQ = 7;        % 128-DSQ carries 7 bits per symbol
numCodewords = 1;      % Simplified: only 1 codeword
numInfoBitsLDPC = k * numCodewords;
totalLDPCInfoBits = numInfoBitsLDPC;

% Sampling and channel parameters
fs = 800e6;            % Sampling rate: 800 MHz
Ts = 1 / fs;           % Sampling period
EbN0_dB = 30;          % Signal-to-noise ratio (dB)

% 16-PAM levels for 128-DSQ mapping
pamLevels = -15:2:15;  % 16 discrete levels: [-15, -13, ..., 13, 15]

% Generate random information bits
txBitsLDPC = randi([0 1], numInfoBitsLDPC, 1);

% Pad bits (no padding needed here since exactly 1 codeword)
txBitsLDPC_padded = txBitsLDPC;

%% 2. LDPC Encoding
G_filename = 'C:\Users\Youan\OneDrive\桌面\Crosstalk\802.3an-2006\matrices\matrices\g.txt';
G = loadGMatrix(G_filename, k, n);
disp(['G Matrix Size: ', num2str(size(G))]);
disp(['Number of Non-Zero Elements: ', num2str(nnz(G))]);
codedBits = [];
for i = 1:numCodewords
    infoBlock = txBitsLDPC_padded((i-1)*k+1 : i*k);
    codeword = mod(infoBlock' * G, 2)';
    codedBits = [codedBits; codeword];
end

%% 3. 128-DSQ Modulation
txSymbols = dsq128Mapper(codedBits, bitsPerDSQ, pamLevels);
% txSymbols is an N x 2 matrix, each row is a DSQ symbol pair

% Form complex baseband signal
txSignal = txSymbols(:,1) + 1j * txSymbols(:,2);

% Test modulation and demodulation
testBits = codedBits(1:700); % Take first 100 symbols' bits (100 * 7 = 700)
testSymbols = dsq128Mapper(testBits, bitsPerDSQ, pamLevels);
testRecoveredBits = dsq128Demapper(testSymbols, pamLevels);
testBER = biterr(testBits, testRecoveredBits) / length(testBits);
fprintf('Modulation-Demodulation Test BER: %.2e\n', testBER);

%% 4. Channel Modeling
% Simplified: only 2 channels (main channel h11 and crosstalk channel h12)
channelOrder = 10; % Simplified channel order
h11 = fir2(channelOrder, [0 1], [1 0.1]); h11 = h11 / norm(h11); % Main channel
h12 = 0.5 * fir2(channelOrder, [0 1], [1 0.05]); h12 = h12 / norm(h12); % Crosstalk channel

% Transmitted signal for the other channel (placeholder)
txSignal2 = txSignal; % Channel 2 signal

%% 5. MISO Pre-compensation Model
% Compute pre-compensation filter l(1,2) = H12 / H11 in frequency domain
fftLen = 512; % Simplified FFT length
freqResp_h11 = fft(h11, fftLen);
freqResp_h12 = fft(h12, fftLen);

% Add regularization to avoid division by small values
epsilon = 1e-6;
l12_freq = freqResp_h12 ./ (freqResp_h11 + epsilon);
l12 = real(ifft(l12_freq, fftLen));

% Truncate filter
filterLength = 10;
l12 = l12(1:filterLength);

% Apply pre-compensation
u1_miso = txSignal - filter(l12, 1, txSignal2);

% Received signal on channel 1 (MISO model)
y1_miso = filter(h11, 1, u1_miso) + filter(h12, 1, txSignal2);

% Add AWGN
signalPower_miso = var(y1_miso);
noisePower_miso = signalPower_miso / (10^(EbN0_dB / 10));
thermalNoise_miso = sqrt(noisePower_miso) * (randn(length(y1_miso), 1) + 1j * randn(length(y1_miso), 1)) / sqrt(2);
y1_miso = y1_miso + thermalNoise_miso;

%% 6. Standard Model (No MISO Pre-compensation)
% Transmit directly through the channel
u1_std = txSignal; % No pre-compensation
y1_std = filter(h11, 1, u1_std) + filter(h12, 1, txSignal2);

% Add AWGN
signalPower_std = var(y1_std);
noisePower_std = signalPower_std / (10^(EbN0_dB / 10));
thermalNoise_std = sqrt(noisePower_std) * (randn(length(y1_std), 1) + 1j * randn(length(y1_std), 1)) / sqrt(2);
y1_std = y1_std + thermalNoise_std;

%% 7. Receiver-side FFE Equalization (Simplified)
FFE_order = 5; % Simplified FFE order
L_f = FFE_order + 1;
H_conv = convmtx(h11, L_f); % Convolution matrix for main channel

% Desired response (delayed delta)
[~, peak_idx] = max(abs(h11));
delay = peak_idx + floor(FFE_order / 2);
if delay > size(H_conv, 1)
    delay = size(H_conv, 1);
end
d_vec = zeros(size(H_conv, 1), 1);
d_vec(delay) = 1;

% Zero-Forcing FFE coefficients
f = pinv(H_conv) * d_vec;

% Apply FFE
ffeOutput_miso = filter(f, 1, y1_miso);
ffeOutput_std = filter(f, 1, y1_std);

%% 8. DSQ Demapping and Bit Recovery
% MISO Model
numSymbols_miso = floor(length(ffeOutput_miso) / 2);
ffeOutputPairs_miso = reshape(ffeOutput_miso(1:2*numSymbols_miso), 2, numSymbols_miso).';
rxBits_miso = dsq128Demapper(ffeOutputPairs_miso, pamLevels);

% Standard Model
numSymbols_std = floor(length(ffeOutput_std) / 2);
ffeOutputPairs_std = reshape(ffeOutput_std(1:2*numSymbols_std), 2, numSymbols_std).';
rxBits_std = dsq128Demapper(ffeOutputPairs_std, pamLevels);

% Truncate to original length
minLen_miso = min(length(rxBits_miso), numInfoBitsLDPC);
rxDecodedBits_miso = rxBits_miso(1:minLen_miso);

minLen_std = min(length(rxBits_std), numInfoBitsLDPC);
rxDecodedBits_std = rxBits_std(1:minLen_std);

%% 9. Performance Evaluation
% MISO Model
[errCount_miso, ber_miso] = biterr(rxDecodedBits_miso, txBitsLDPC(1:minLen_miso));
fprintf('MISO Model BER: %.2e (Errors: %d, Bits: %d)\n', ber_miso, errCount_miso, minLen_miso);

ffeDelay = delay - 1;
if ffeDelay > 0
    txSignalAligned_miso = txSignal(ffeDelay+1:end);
    ffeOutputTrimmed_miso = ffeOutput_miso(1:length(txSignalAligned_miso));
else
    txSignalAligned_miso = txSignal(1:length(ffeOutput_miso));
    ffeOutputTrimmed_miso = ffeOutput_miso;
end
errorSignal_miso = ffeOutputTrimmed_miso - txSignalAligned_miso;
signalPower_miso = mean(abs(txSignalAligned_miso).^2);
noisePower_miso = mean(abs(errorSignal_miso).^2);
dp_snr_miso = 10 * log10(signalPower_miso / noisePower_miso);
fprintf('MISO Model DP-SNR: %.2f dB\n', dp_snr_miso);

% Standard Model
[errCount_std, ber_std] = biterr(rxDecodedBits_std, txBitsLDPC(1:minLen_std));
fprintf('Standard Model BER: %.2e (Errors: %d, Bits: %d)\n', ber_std, errCount_std, minLen_std);

if ffeDelay > 0
    txSignalAligned_std = txSignal(ffeDelay+1:end);
    ffeOutputTrimmed_std = ffeOutput_std(1:length(txSignalAligned_std));
else
    txSignalAligned_std = txSignal(1:length(ffeOutput_std));
    ffeOutputTrimmed_std = ffeOutput_std;
end
errorSignal_std = ffeOutputTrimmed_std - txSignalAligned_std;
signalPower_std = mean(abs(txSignalAligned_std).^2);
noisePower_std = mean(abs(errorSignal_std).^2);
dp_snr_std = 10 * log10(signalPower_std / noisePower_std);
fprintf('Standard Model DP-SNR: %.2f dB\n', dp_snr_std);

%% 10. Visualization
% Constellation Diagrams
figure;
subplot(1, 2, 1);
scatter(real(ffeOutputPairs_miso(:,1)), real(ffeOutputPairs_miso(:,2)), '.', 'DisplayName', 'MISO Received');
hold on;
scatter(txSymbols(:,1), txSymbols(:,2), 'ro', 'filled', 'DisplayName', 'Ideal');
title('MISO Model 128-DSQ Constellation');
xlabel('PAM16 Dimension 1');
ylabel('PAM16 Dimension 2');
legend;
grid on;

subplot(1, 2, 2);
scatter(real(ffeOutputPairs_std(:,1)), real(ffeOutputPairs_std(:,2)), '.', 'DisplayName', 'Standard Received');
hold on;
scatter(txSymbols(:,1), txSymbols(:,2), 'ro', 'filled', 'DisplayName', 'Ideal');
title('Standard Model 128-DSQ Constellation');
xlabel('PAM16 Dimension 1');
ylabel('PAM16 Dimension 2');
legend;
grid on;

% Signal Waveforms (Real Part)
t = (0:length(txSignal)-1) * Ts;
figure;
subplot(2, 1, 1);
plot(t, real(y1_miso), 'b-', 'DisplayName', 'MISO Received (Real)');
hold on;
plot(t, real(filter(h11, 1, txSignal)), 'r--', 'DisplayName', 'Ideal (Real)');
title('MISO Model Received Signal Waveform');
xlabel('Time (s)');
ylabel('Amplitude');
legend;
grid on;

subplot(2, 1, 2);
plot(t, real(y1_std), 'b-', 'DisplayName', 'Standard Received (Real)');
hold on;
plot(t, real(filter(h11, 1, txSignal)), 'r--', 'DisplayName', 'Ideal (Real)');
title('Standard Model Received Signal Waveform');
xlabel('Time (s)');
ylabel('Amplitude');
legend;
grid on;

disp('Simplified Simulation Completed!');

%% Function Definitions
function G = loadGMatrix(filename, numRows, numCols)
    % Open file
    fid = fopen(filename, 'r');
    if fid == -1
        error('Cannot open file: %s', filename);
    end
    
    % Read all lines
    G_lines = textscan(fid, '%s', 'Delimiter', '\n');
    fclose(fid);
    G_lines = G_lines{1};
    
    % Initialize sparse matrix row and column indices
    row_indices = [];
    col_indices = [];
    
    % Parse each line
    for i = 1:numRows
        line = strtrim(G_lines{i});
        if ~isempty(line)
            % Parse column indices (0-based)
            indices = str2num(line); %#ok<ST2NM>
            % Convert to 1-based indices
            col_indices = [col_indices, indices + 1]; %#ok<AGROW>
            row_indices = [row_indices, i * ones(1, length(indices))]; %#ok<AGROW>
        end
    end
    
    % Build sparse matrix
    G = sparse(row_indices, col_indices, 1, numRows, numCols);
    
    % Validate matrix
    fprintf('G Matrix Dimensions: %d×%d\n', size(G, 1), size(G, 2));
    fprintf('Number of Non-Zero Elements: %d\n', nnz(G));
    if nnz(G) == 0
        warning('G Matrix is empty, please check file content or path: %s', filename);
    end
end

function mappedSymbols = dsq128Mapper(codedBits, bitsPerDSQ, pamLevels)
    % Ensure pamLevels is a row vector
    pamLevels = pamLevels(:)';
    
    % Calculate number of symbols
    numSymbols = floor(length(codedBits) / bitsPerDSQ);
    
    % Truncate excess bits to align with symbol boundaries
    codedBits = codedBits(1:numSymbols * bitsPerDSQ);
    
    % Reshape bits into numSymbols x bitsPerDSQ matrix
    bitMatrix = reshape(codedBits, bitsPerDSQ, numSymbols)';
    
    % Initialize output symbol matrix
    mappedSymbols = zeros(numSymbols, 2);
    
    % Gray code mapping (4 bits to 16-PAM)
    grayMap = [0 1 3 2 6 7 5 4 12 13 15 14 10 11 9 8]; % Gray code order
    pamIndices = 0:15;
    pamMapping = zeros(1, 16);
    for i = 0:15
        pamMapping(grayMap(i+1)+1) = pamIndices(i+1);
    end
    
    % Map bits to symbols
    for i = 1:numSymbols
        bits = bitMatrix(i, :);
        
        % Map first 4 bits to I component (16-PAM)
        I_bits = bits(1:4);
        I_val = bi2de(I_bits, 'left-msb'); % Convert to decimal (0 to 15)
        I_val = pamMapping(I_val + 1); % Apply Gray code mapping
        mappedSymbols(i, 1) = pamLevels(I_val + 1); % Map to PAM level
        
        % Map bits [1, 5, 6, 7] to Q component (16-PAM)
        Q_bits = [bits(1), bits(5:7)];
        Q_val = bi2de(Q_bits, 'left-msb'); % Convert to decimal (0 to 15)
        Q_val = pamMapping(Q_val + 1); % Apply Gray code mapping
        mappedSymbols(i, 2) = pamLevels(Q_val + 1); % Map to PAM level
    end
end

function rxBits = dsq128Demapper(symbols, pamLevels)
    % Number of symbols
    numSymbols = size(symbols, 1);
    bitsPerDSQ = 7;
    rxBits = zeros(numSymbols * bitsPerDSQ, 1);
    
    % Gray code mapping (4 bits to 16-PAM)
    grayMap = [0 1 3 2 6 7 5 4 12 13 15 14 10 11 9 8]; % Gray code order
    invGrayMap = zeros(1, 16);
    for i = 0:15
        invGrayMap(grayMap(i+1)+1) = i;
    end
    
    % Process each symbol pair
    for i = 1:numSymbols
        % Find closest PAM level indices for I and Q
        [~, I_idx] = min(abs(pamLevels - symbols(i, 1)));
        [~, Q_idx] = min(abs(pamLevels - symbols(i, 2)));
        
        % Convert to 0-based indices
        I_val = I_idx - 1; % 0 to 15
        Q_val = Q_idx - 1; % 0 to 15
        
        % Inverse Gray code mapping
        I_val = invGrayMap(I_val + 1);
        Q_val = invGrayMap(Q_val + 1);
        
        % Convert indices to 4-bit binary
        I_bits = de2bi(I_val, 4, 'left-msb');
        Q_bits = de2bi(Q_val, 4, 'left-msb');
        
        % Reconstruct 7-bit sequence
        bits = [I_bits(1); I_bits(2:4)'; Q_bits(2:4)'];
        
        % Store bits
        rxBits((i-1)*bitsPerDSQ + 1 : i*bitsPerDSQ) = bits;
    end
end