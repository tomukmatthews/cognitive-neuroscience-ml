
% Author: Tom Matthews, Part III Cambridge Physics student

% This script partitions a dataset into 45s epochs, then extracts the
% reactions times, subjective drowsiness scores. The script then calculates
% a series of features for each epoch, and removes epochs which contain no
% reaction times. The variable signf_thresh can be set such that only
% features displaying correlations with reaction times below a certain
% P-value are retained (setting to 1 retains all features). All of the
% CDcount variables and if statements are used to keep track of which
% features are included from which sector. A number of feature selection
% algorithms are tested at the bottom.

%% Prepare the pre-processed data for analysis
dataset = 15;
% Global variables
sample_entropy = false;
Occipital = true;
epochLen = 45;     % Epoch length in seconds
signf_thresh = 1;
freq_upperlim = 40; % Frequency upper limit must be even

pathappend = '/Users/tommatthews/part3project/';
eeglab_toolbox = [pathappend 'toolboxes/eeglab14_1_2b'];
ftp_toolbox = [pathappend 'toolboxes/fieldtrip-20190120'];
scripts = [pathappend 'scripts'];
addpath(scripts)
addpath(genpath(eeglab_toolbox));
addpath(ftp_toolbox);
rmpath(genpath([eeglab_toolbox '/functions/octavefunc']));
S.eeg_filepath = [ pathappend 'wipers_data_final'];
S.eeg_filename = ['P' num2str(dataset) '_final_flt'];
evalexp = 'pop_loadset(''filename'', [S.eeg_filename ''.set''], ''filepath'', S.eeg_filepath);';
[T,EEG] = evalc(evalexp);

% Now epoch the data
EEG = eeg_regepochs(EEG, epochLen, [0 epochLen], NaN);
EEG_coh = EEG;

% Select occipital electrodes, or  use all electrodes, could try 
% using parietal as well.
if Occipital
    chanlabels = {EEG.chanlocs.labels};
%     electrodes_occ = {'Oz','O1','O2'};
    electrodes_occ = {'Pz','P1','P2'};
    selec_elec = ismember(chanlabels,electrodes_occ);
    remove_elec = find(~selec_elec);
    %Use only selected electrodes
    evalexp = 'pop_select(EEG,''nochannel'',remove_elec);';
    [T,EEG_occ] = evalc(evalexp);
    EEG = EEG_occ;
end

% Call function to find reaction times and subjective drowsiness scores,
% along with their latencies
[reaction_times,time_points,subj_dscore,time_dscore,time,Resp_lm,RT_lm,RT_lstd,id] = identify_alertness(EEG);

%% Feature extraction: BLINKS -- PSD -- ENTROPY -- COHERENCE

% Create the an array to store the features and epoch target class labels.
class_data = [];
ft_names = [];
CDcount1 = 1;
CDcount2 = 1;

% Extract features associated with blinks - duration, AVRB etc...
[blinks_CD,id] = blinks(EEG,RT_lm,id,pathappend,dataset,epochLen,signf_thresh);
class_data = [class_data blinks_CD];

if isempty(blinks_CD) == 0
    CDcount2 = size(class_data,2);
    disp(['Blink Features: ' num2str(CDcount1) ' - ' num2str(CDcount2)])
    CDcount1 = CDcount2+1;
end

time(id)=[];
RT_lm(id)=[];
RT_lstd(id)=[];
Resp_lm(id)=[];
N_epch = length(time);
RT_lm = RT_lm ./ (Resp_lm .^ 0.5); % Rescale reaction times to reflect missed trials

% Extract features assosciated with the Power Spectral Dennsity
disp('Calculating PSD features...')
[SpectralPower ,SpectralCentroid, SpectralVariance, SE, DPF, DPP]...
                              = classify_computePowerfts(EEG,id,epochLen,time,RT_lm);
                          
% Find the sample entropy of each epoch for one electrode.
if sample_entropy
    disp('Calculating Sample Entropy feature...')
    SmpEntropy = SampleEntropy(EEG,id,RT_lm,signf_thresh);
end

% Plot correlation between power and local mean reaction time.
freq_upperlim = 40;
freqIdx = 0:2:freq_upperlim;

PSD1 = [];
PSD2 = [];

for idx = 1:freq_upperlim/2
    
%     disp([num2str(freqIdx(idx:idx+1)),' correlation coefficients']);
%     disp('Power');
    [RHO_power,PVAL_power] = corr(RT_lm(:,1),SpectralPower(:,idx),'Type','Spearman');
        
    correl_array1(idx) = RHO_power;
    f_array = (freqIdx(2:end)-1)';
        
    if PVAL_power < signf_thresh
        PSD1 = [PSD1 SpectralPower(:,idx)];
    end
   
end

class_data = [class_data PSD1];

if isempty(PSD1) == 0
    CDcount2 = size(class_data,2);
    disp(['2 Hz Bin Features: ' num2str(CDcount1) ' - ' num2str(CDcount2)])
    CDcount1 = CDcount2+1;
end

PSD2tmp = [SpectralPower(:,21:25) SpectralCentroid SpectralVariance DPF DPP SE];

for idx = 1:size(PSD2tmp,2)
    [RHO,PVAL] = corr(RT_lm(:,1),PSD2tmp(:,idx),'Type','Spearman');
    if PVAL < signf_thresh
        PSD2 = [PSD2 PSD2tmp(:,idx)];
    end
end

class_data = [class_data PSD2];

if isempty(PSD2) == 0
    CDcount2 = size(class_data,2);
    disp(['Classical Band Features: ' num2str(CDcount1) ' - ' num2str(CDcount2)])
    CDcount1 = CDcount2+1;
end

if sample_entropy
    Entropy = SmpEntropy;
    class_data = [class_data Entropy];
    
    if isempty(Entropy) == 0
        CDcount2 = size(class_data,2);
        disp(['Sample Entropy Features: ' num2str(CDcount2)])
        CDcount1 = CDcount2+1;
    end
end

plot_power = true;
if plot_power
    figure
    hold on
    plot(f_array,correl_array1)
    yline(-0.25,'--');
    yline(0,'-');
    yline(0.25,'--');
    hold off
    ylim([-0.7 0.7])
    legend('Relative Power')
    title('Relative power - RT correlations vs frequency')
    xlabel('Frequency / Hz')
    ylabel('Correlation Coefficient')
end

% Next calculate the coherence features
eleclabels.frontal = {'F7', 'F8', 'Fz'};
eleclabels.central = {'C3', 'C4'};
eleclabels.parietal = {'Pz'};
eleclabels.temporal =  {'T7', 'T8'};
eleclabels.occipital = {'Oz','O1', 'O2'};
disp('Calculating Coherence features...')
[coh] = classify_computeCoherencefts(EEG_coh,eleclabels);
coh_features = table2array(coh);
coh_features(id,:)=[];
disp('\Coherence features calculated\')

coh_array = [];
disp('Now for the coherence correlations');
for idx = 1:size(coh_features,2)
%     disp([num2str(idx),' correlation coefficients']);
    [RHO,PVAL] = corr(RT_lm(:,1),coh_features(:,idx),'Type','Spearman');
    if PVAL < signf_thresh
        coh_array = [coh_array coh_features(:,idx)];
    end
end    

% Update feature array
class_data = [class_data coh_array];

if isempty(coh_array) == 0
    CDcount2 = size(class_data,2);
    disp(['Coherence Features: ' num2str(CDcount1) ' - ' num2str(CDcount2)])
end

% Collect the feature matrix and target output (RT's) as one matrix for
% subsequent analysis (collecting before and after soft normalisation).
CD = class_data;
CD = [CD RT_lm];
% Next perform 'soft' normalisation on the feature vectors
mn_cd = mean(class_data);
std_cd = std(class_data);
class_data = (class_data - mn_cd)./(std_cd);
predictors = class_data;
class_data = [class_data RT_lm];


%% FEATURE SELECTION

disp('SEQUENTIAL FEATURE SELECTION')
ds = size(class_data,1);
perm_ind = randperm(ds);
class_data_mix = class_data(perm_ind,:);
xtrain = class_data_mix(1:floor(0.7*ds),1:end-1);
xtest = class_data_mix(floor(0.7*ds)+1:end,1:end-1);
ytrain = class_data_mix(1:floor(0.7*ds),end);
ytest = class_data_mix(floor(0.7*ds)+1:end,end);
% ypred = classify(xtest, xtrain, ytrain);
% fun = @(xtrain, ytrain, xtest, ytest) sum(ytest ~= classify(xtest, xtrain, ytrain));
fun = @(xtrain,ytrain,xtest,ytest) norm(ytest-xtest*(xtrain\ytrain))^2;
opts = statset('display','iter');
[in,history] = sequentialfs(fun,class_data(:,1:end-1),class_data(:,end),...
    'cv',5,'options',opts,'direction','forward');


disp('STEPWISE FIT')
nrows = size(class_data,1);
stepwisefit([ones(nrows,1) class_data(:,1:end-1)],RT_lm)

disp('NCA FIT')
ds = size(class_data,1);
perm_ind = randperm(ds);
class_data_mix = class_data(perm_ind,:);
Xtrain = class_data_mix(1:floor(0.7*ds),1:end-1);
Xtest = class_data_mix(floor(0.7*ds)+1:end,1:end-1);
ytrain = class_data_mix(1:floor(0.7*ds),end);
ytest = class_data_mix(floor(0.7*ds)+1:end,end);
nca = fsrnca(Xtrain,ytrain,'FitMethod','exact', 'Solver','sgd');
figure
plot(nca.FeatureWeights,'ro')
xlabel('Feature index')
ylabel('Feature weight')
grid on
L = loss(nca,Xtest,ytest);

figure
title(['Dataset ' num2str(dataset)])
hold on
scatter(time,RT_lm)
plot(time_dscore,subj_dscore)
xlabel('Time / s')
ylabel('RT / s')

