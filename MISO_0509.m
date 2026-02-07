%% ================================================================
%  10GBASE‑T  Legacy (No Pre-compensation)  vs.  Proposed (MISO Pre-compensation + FFE)
%  -- Full-Link Monte-Carlo Calculation of FEXT Suppression / DP‑SNR / BER  --
% ================================================================

%% ---------- Parameter Section ----------
Nsym     = 3e4;      % Number of symbols (Training + Validation)
trainSym = 4e3;      % Training length
SNR_dB   = 20;       % AWGN Signal-to-Noise Ratio (Increase -> Less Noise; 18–25 dB acceptable)
P_FEXTdB = -25;      % Typical ELFEXT for a single Cat‑6A 55 m cable (-25 dB more challenging)
tapFFE   = 3;        % Number of taps for Rx-side FFE
muTx     = 2e-3;     % LMS step size at Tx side
muRx     = 5e-3;     % LMS step size at Rx side
rng(42);             % Fix random seed for reproducibility
%% ---------------------------------------------------------------

M     = 16;                                 % PAM‑16 (1D equivalent of DSQ128)
pamSym= @(n) (randi([-(M/2-1) (M/2-1)], n,1)*2+1);   % Generate PAM16
sigma = 10^(-SNR_dB/20);                    % AWGN σ
alpha = 10^(P_FEXTdB/20);                   % FEXT linear amplitude

%% ---------- 4×4 Channel Matrix ----------
Hdir       = eye(4);                        % Direct channel
Hfext      = alpha * (randn(4) - 0.5);      % Pure real random coupling
Hfext(1:5:end) = 0;                         % Zero the diagonal
H          = Hdir + Hfext;                  % Total 4×4

%% ---------- Generate Transmit Symbols ----------
txSym = pamSym(Nsym*4);  tx = reshape(txSym, Nsym,4);   % Nsym×4

%% ---------- Legacy: No Pre-compensation ----------
noiseL = sigma*randn(Nsym,4);
rxLegacy = tx*H.' + noiseL;                 % Direct + noise

%% ---------- Proposed: MISO Pre-compensation + FFE ----------
% Initial MISO coefficients: Directly take Hfext(1,2:4)
Lpre = zeros(1,4);    Lpre(2:4) = Hfext(1,2:4);
% Rx FFE coefficients
Wffe = zeros(1,tapFFE);
yHist = zeros(1,tapFFE-1);      % History register
z_hat = zeros(Nsym,1);          % Store equalizer output

for k = 1:Nsym
    %------ Transmit-side Pre-compensation (only for pair‑1) ------
    u = tx(k,:);                                      % 4‑element
    u(1) = tx(k,1) - tx(k,2:4)*Lpre(2:4).';           % MISO linear compensation

    %------ Channel ------
    y = u * H.' + sigma*randn(1,4);                   % Rx real signal
    yi = y(1);                                        % Focus on pair 1

    %------ Rx FFE ------
    yvec = [yi yHist];                                % tapFFE‑dimensional
    zi   = Wffe * yvec.';                             % FFE output
    d    = tx(k,1);                                   % Training target
    err  = d - zi;

    %------ LMS Update (Training Phase) ------
    if k <= trainSym
        Wffe = Wffe + muRx * err * yvec;
        Lpre(2:4) = Lpre(2:4) - muTx * err * tx(k,2:4);
    end

    %------ Update Register & Save Output ------
    yHist = [yi yHist(1:end-1)];
    z_hat(k) = zi;
end

%% ---------- Statistical Metrics ----------
% 1) FEXT Suppression  = Original FEXT Power / (Residual FEXT Power)
origFEXT   = tx(:,2:4)*Hfext(1,2:4).';               % Legacy FEXT component
predFEXT   = tx(:,2:4)*Lpre(2:4).';                  % Pre-compensation cancellation amount
resiFEXT   = origFEXT - predFEXT;                    % Residual
FEXTsupp_dB= 10*(1 + log10( mean(origFEXT.^2) / mean(resiFEXT.^2) ) );

% 2) DP‑SNR (calculated from validation segment)
testIdx  = trainSym+1 : Nsym;
dpSNR_L  = 10*log10( var(tx(testIdx,1)) / var(rxLegacy(testIdx,1)-tx(testIdx,1)) );
dpSNR_P  = 10*log10( var(tx(testIdx,1)) / var(z_hat(testIdx)-tx(testIdx,1)) );

% 3) BER (simple hard decision)
slice = @(r) sign(r).*(2*round((abs(r)-1)/2)+1);
berL = mean( slice(rxLegacy(testIdx,1)) ~= tx(testIdx,1) );
berP = mean( slice(z_hat(testIdx))      ~= tx(testIdx,1) );

%% ---------- Print Results ----------
fprintf('\n=== Simulation summary ===\n');
fprintf('FEXT suppression        : %.2f  dB\n', FEXTsupp_dB);
fprintf('DP-SNR  Legacy          : %.2f  dB\n', dpSNR_L);
fprintf('DP-SNR  Proposed        : %.2f  dB\n', dpSNR_P);
fprintf('BER     Legacy          : %.3e\n', berL);
fprintf('BER     Proposed        : %.3e\n', berP);

%% ---------- Plot ----------
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


%% =============== Added Section: Complexity & Latency =================
% ---- Legacy DSP tap configuration (adjust numbers if implementations differ) ----
tapTHP_legacy   = 48;        % Tx-side THP FIR taps / pair
tapFFE_legacy   = 64;        % Rx-side FFE taps / pair
tapDFE_legacy   = 128;       % Rx-side DFE taps / pair

tapMISO_prop    = 12;        % Proposed: MISO FIR taps / pair  (Example)
% tapFFE (given in parameter section above)  = Proposed FFE taps / pair

% ---- Multiplier Count (sum linearly for four pairs in parallel) ----
mulLegacy  = 4*(tapTHP_legacy + tapFFE_legacy + tapDFE_legacy);
mulPropose = 4*(tapMISO_prop  + tapFFE);

% ---- Algorithm-level Pipeline Delay (taps approximate UI) ----
latLegacy  = (tapTHP_legacy + tapFFE_legacy + tapDFE_legacy);
latPropose = (tapMISO_prop  + tapFFE);

% ---- Print ----
fprintf('--- Complexity & Latency -----------------------------\n');
fprintf('Multipliers Legacy | Proposed : %d | %d  (%.0f %% fewer)\n',...
        mulLegacy, mulPropose, 100*(1-mulPropose/mulLegacy));
fprintf('Latency    Legacy | Proposed : %d | %d  UI  (%.0f %% shorter)\n',...
        latLegacy, latPropose, 100*(1-latPropose/latLegacy));

% ---- Plot Second Figure ----
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


SNR_vec = 14:2:26;          % 7 test points
berL = zeros(size(SNR_vec));
berP = zeros(size(SNR_vec));

for s = 1:numel(SNR_vec)
    SNR_dB = SNR_vec(s);
    sigma  = 10^(-SNR_dB/20);
    ...    % Keep the rest of the code, but don't open figures inside the loop
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
    ...     % Keep SNR=20 dB, run once
    berTap(t)  = mean(slice(z_hat(testIdx))~=tx(testIdx,1));
    mulTap(t)  = 4*(tapMISO+tapFFE);         % Multiplier count
end

figure;
yyaxis left; semilogy(tapList, berTap,'-o'); ylabel('BER');
yyaxis right; plot(tapList, mulTap,'--s');   ylabel('Multipliers');
grid on; xlabel('MISO tap count');
title('Performance–Complexity trade‑off (Proposed chain)');
legend({'BER','Real mults'},'Location','east');