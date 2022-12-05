% configure the epochs
n = 200;
srate = 128;
prestim = 200;
length = 1200;
montage = 'BioSemi64';
config = struct('n', n, 'srate', srate, 'length', length, 'prestim', prestim);

% obtain 64 source locations
lf = lf_generate_fromnyhead('montage', montage);
sources = lf_get_source_spaced(lf, 64, 25);

% assign a signal to a subset of the sources and simulating data
sigcomps70 = utl_get_component_fromtemplate('visual_n70_erp', lf);
sigcomps100 = utl_get_component_fromtemplate('visual_p100_erp', lf);
sigcomps135 = utl_get_component_fromtemplate('visual_n135_erp', lf);
sigcomps300 = utl_get_component_fromtemplate('p300_erp', lf);
components = [sigcomps70 sigcomps100 sigcomps135 sigcomps300];
for i=1:max(size(components))
    components(i).signal{1,1}.peakLatency = components(i).signal{1,1}.peakLatency + prestim;
    components(i).signal{1,1}.peakLatencyDv = components(i).signal{1,1}.peakLatencyDv * 1.5;
end
sigdata = generate_scalpdata(components, lf, config);
sigdata = sigdata * 10; % adjust scale

% create undistorted eeg dataset
EEG_raw = utl_create_eeglabdataset(sigdata, config, lf);
pop_saveset(EEG_raw, 'filename', 'EEGLAB_raw.set');

% assign an environmental noise to the data
noiseact = struct('type', 'noise', 'color', 'pink', 'amplitude', 1);
noisecomps = utl_create_component(sources, noiseact, lf);
noisedata = generate_scalpdata(noisecomps, lf, config);

% mix signal and noise
division_factor = 0.15;
batch_size = round(division_factor*n);
easy_distorted = sigdata(:,:,1:batch_size) + 10 * noisedata(:,:,1:batch_size);
heavy_distorted = sigdata(:,:,batch_size:end) + 15 * noisedata(:,:,batch_size:end);
distorted = cat(3, easy_distorted, heavy_distorted);

% add impulsive noise as Bernoulli-Gaussian sequence
rng(23)
lambda = 0.1;
variance = 5;
bernoulli_sequence = rand(size(distorted(:, :, batch_size:(end-batch_size)))) <= lambda;
imp_noise = randn(size(distorted(:, :, batch_size:(end-batch_size)))) .* bernoulli_sequence;
distorted(:, :, batch_size:(end-batch_size)) = distorted(:, :, batch_size:(end-batch_size)) + sqrt(variance) * imp_noise;

% add impulsive noise as signed Bernoulli sequence
lambda = 0.002;
amplitude = 50;
bernoulli_sequence = rand(size(distorted(:, :, (end-2*batch_size):(end-batch_size)))) <= lambda;
sign_alternation = (rand(size(distorted(:, :, (end-2*batch_size):(end-batch_size)))) > 0.5)*2 - 1;
imp_noise = sign_alternation .* bernoulli_sequence;
distorted(:, :, (end-2*batch_size):(end-batch_size)) = distorted(:, :, (end-2*batch_size):(end-batch_size)) + amplitude * imp_noise;

selected_epochs = datasample(distorted, batch_size, 3, 'Replace', false);
% add inverted epochs
distorted = cat(3, distorted, -selected_epochs);
% add corrupted epochs
distorted = cat(3, distorted, 2*randn(size(selected_epochs)));

EEG_distorted = utl_create_eeglabdataset(distorted, config, lf);
pop_saveset(EEG_distorted, 'filename', 'EEGLAB_distorted.set');

