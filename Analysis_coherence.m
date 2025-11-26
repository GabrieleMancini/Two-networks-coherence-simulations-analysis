clear all
close all

exp_lab = 'pv';

%% Import Data
FR_1 = readmatrix('firing_rate_trace_trial1.csv');
FR_2 = readmatrix('firing_rate_trace_trial2.csv');
FR_3 = readmatrix('firing_rate_trace_trial3.csv');
FR_4 = readmatrix('firing_rate_trace_trial4.csv');
FR_5 = readmatrix('firing_rate_trace_trial5.csv');

FR_E_1 = [FR_1(:,1) FR_2(:,1) FR_3(:,1) FR_4(:,1) FR_5(:,1)]';
FR_I_1 = [FR_1(:,2) FR_2(:,2) FR_3(:,2) FR_4(:,2) FR_5(:,2)]'; 
FR_E_2 = [FR_1(:,3) FR_2(:,3) FR_3(:,3) FR_4(:,3) FR_5(:,3)]'; 
FR_I_2 = [FR_1(:,4) FR_2(:,4) FR_3(:,4) FR_4(:,4) FR_5(:,4)]'; 

LFP_pop_1 = readmatrix('LFP_pop_1.csv');
LFP_pop_2 = readmatrix('LFP_pop_2.csv');

%% Preprocessing
LFP_1 = LFP_pop_1(2:end-1, 501:end);
LFP_2 = LFP_pop_2(2:end-1, 501:end);

fs = 1000;
nRows = size(LFP_1, 1);
chunkLength_s = 10;
chunkLength = fs * chunkLength_s; 
nChunks = floor(size(LFP_1,2) / chunkLength);

% Chunk LFPs
for i = 1:nRows
    for c = 1:nChunks
        idx = (c-1)*chunkLength + 1 : c*chunkLength;
        LFP_1_chunks{i}(c,:) = LFP_1(i, idx);
        LFP_2_chunks{i}(c,:) = LFP_2(i, idx);
    end
end

%% Frequency Bands
bands = struct( ...
    'slow', [0.1 1], ...
    'delta', [1 4], ...
    'theta', [4 8], ...
    'alpha', [8 13], ...
    'beta',  [13 30]);

bandNames = fieldnames(bands);
nBands = numel(bandNames);

%% Compute Coherence
for i = 1:nRows
    signal1 = LFP_1_chunks{i};
    signal2 = LFP_2_chunks{i};

    for b = 1:nBands
        range = bands.(bandNames{b});
        f_center = mean(range);
        nCycles = 2;
        winLength = round(max(2, nCycles / f_center) * fs);
        window = hamming(winLength);
        noverlap = round(0.5 * winLength);
        nfft = winLength;

        for trial = 1:nChunks
            freqs = 0.1:0.5:(fs/2);
            [Cxy, f] = mscohere(signal1(trial,:), signal2(trial,:), window, noverlap, freqs, fs);

            if b ==1
                full_coherence_all{i,trial} = Cxy;
                full_freq_all{i,trial} = f';
            end

            band_idx = f >= range(1) & f <= range(2);
            coherence_all{i}(b, trial) = mean(Cxy(band_idx));
        end
    end
end

%% Aggregate Statistics
for i = 1:nRows
    for b = 1:nBands
        coh_mean_by_band(i,b) = mean(coherence_all{i}(b,:));
        coh_std_by_band(i,b) = std(coherence_all{i}(b,:))/sqrt(nChunks-1);
    end
end

%% Statistical analysis
for b = 1:nBands
    [p_values(b),~,stats] = ranksum(coherence_all{1}(b,:), coherence_all{nRows}(b,:));
end

%% Compute PSD for both populations
PSD_1_all = cell(nRows,1);
PSD_2_all = cell(nRows,1);

for i = 1:nRows
    nTrials = size(LFP_1_chunks{i},1);
    for t = 1:nTrials
        [Pxx1, f_psd] = pwelch(detrend(LFP_1_chunks{i}(t,:)), hamming(2*fs), fs, [], fs);
        [Pxx2, ~]     = pwelch(detrend(LFP_2_chunks{i}(t,:)), hamming(2*fs), fs, [], fs);
        PSD_1_all{i}(t,:) = 10*log10(Pxx1);
        PSD_2_all{i}(t,:) = 10*log10(Pxx2);
    end
end

PSD_1_mean = mean(PSD_1_all{1},1);
PSD_1_sem  = std(PSD_1_all{1},[],1)/sqrt(size(PSD_1_all{1},1));

PSD_2_mean = mean(PSD_1_all{5},1);
PSD_2_sem  = std(PSD_1_all{5},[],1)/sqrt(size(PSD_1_all{5},1));

lin_PSD_1 = 10.^(PSD_1_mean/10);
lin_PSD_2 = 10.^(PSD_2_mean/10);

DOS = (lin_PSD_2 - lin_PSD_1) ./ (lin_PSD_1 + lin_PSD_2);

lin_sem1 = lin_PSD_1 .* log(10)/10 .* PSD_1_sem;
lin_sem2 = lin_PSD_2 .* log(10)/10 .* PSD_2_sem;

DOS_sem = sqrt( ( (2*lin_PSD_2)./(lin_PSD_1+lin_PSD_2).^2 ).^2 .* lin_sem1.^2 + ...
                ( (2*lin_PSD_1)./(lin_PSD_1+lin_PSD_2).^2 ).^2 .* lin_sem2.^2 );

%% SAVE DATA
DATA = struct();
count=1;

for i = 1:4:5
    allCxy = cell2mat(full_coherence_all(i, :)');
    fVals = full_freq_all{i,1};
    meanCxy = movmean(mean(allCxy,1),6);
    stdCxy  = movmean(std(allCxy,[],1)/sqrt(nChunks-1),6);

    DATA.coherence{count} = allCxy;
    DATA.fx{count} = fVals;
    count=count+1;
end

save_path = fullfile('results', ['coherence_DATA_' exp_lab '.mat']);
if ~exist('results','dir'); mkdir('results'); end
save(save_path, 'DATA', '-v7.3');
