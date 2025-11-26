clear all
close all

exp_lab = "hm4di";

% Import Data
path_ctrl = "/Users/gabrielemancini/Desktop/Gozzi Project/Simulations/" + exp_lab + "_control_2";
path_man  = "/Users/gabrielemancini/Desktop/Gozzi Project/Simulations/" + exp_lab + "_manipulation_3";

FR_ctrl_trials = [];
FR_man_trials  = [];

for i = 1:5
    FR_ctrl_trials(:,i) = readmatrix(path_ctrl + "/firing_rate_trace_trial" + i + ".csv")(:,1);
    FR_man_trials(:,i)  = readmatrix(path_man + "/firing_rate_trace_trial" + i + ".csv")(:,1);
end

% Compute population firing rate (weighted)
FR_ctrl = 1.0 * FR_ctrl_trials;  % here 0.8*E +0.2*I simplified as single column
FR_man  = 1.0 * FR_man_trials;

% Chunking
fs = 500;
chunkLength_s = 10;
chunkLength = fs * chunkLength_s;
nChunks = floor(size(FR_man,1)/chunkLength);

FR_ctrl_chunks = [];
FR_man_chunks  = [];

count = 1;
for trial = 1:5
    for c = 1:nChunks
        idx = (c-1)*chunkLength + 1 : c*chunkLength;
        FR_ctrl_chunks(count) = mean(FR_ctrl(idx,trial));
        FR_man_chunks(count)  = mean(FR_man(idx,trial));
        count = count + 1;
    end
end

% Prepare data
data = [FR_ctrl_chunks' FR_man_chunks'];
mean_ctrl = mean(FR_ctrl_chunks);
std_ctrl  = std(FR_ctrl_chunks);
data_z    = (data - mean_ctrl) ./ std_ctrl;

% Save processed data
save_path = fullfile("/Users/gabrielemancini/Desktop/Gozzi Project/Simulations/", "FR_DATA_" + exp_lab + ".mat");
save(save_path, 'data', 'data_z', '-v7.3');
