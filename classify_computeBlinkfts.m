function [blink_CD,id] = blinks(EEG,RT_lm,id,pathappend,dataset,epochLen, signf_thresh)
% blinks:

% Author: Tom Matthews, Part III Cambridge Physics student
% Function to extract the various blink features into an array called
% blink_CD, if signf_thresh is set to below 1 it can be used to selectively
% use blink components according to the Spearmans rank correlation with the
% gold standard. Note that this code requires the use of an EEGLAB plugin
% called 'BLINKER' which was run via the EEGLAB window and outputs an array
% called blinkProp which stores all of the properties of each blink.

% Outputs the blink_CD array containing the feature vectors associated with
% blinks, and the indices of the epochs containing no blinks so that those
% observations can be removed from other features etc.


load([pathappend 'blink_data/P' num2str(dataset) '_ica_blinks.mat']);

S = struct2cell(blinkProperties);
S = squeeze(S);
S = S';
blinkProp = [];
for idx = 1:size(S,2)
    blinkProp = [blinkProp cell2mat(S(:,idx))];
end
% Now remove any rows which have NaNs
[id_row, id_col] = find(isnan(blinkProp));
blinkProp(id_row,:)=[];

% % INPUT
% blinkProp has 25 columns in the following order:
% [1-durationBase, 2-durationZero, 3-durationTent, 4-durationHalfBase,
% 5-durationHalfZero, 6-interblinkMaxAmp, 7-interblinkMaxVelBase,
% 8-interblinkMaxVelZero, 9-negAmpVelRatioBase, 10-posAmpVelRatioBase,
% 11-negAmpVelRatioZero, 12-posAmpVelRatioZero, 13-negAmpVelRatioTent,
% 14-posAmpVelRatioTent, 15-timeShutBase, 16-timeShutzero, 17-timeShutTent,
% 18-closingTimeZero, 19-reopeningTimeZero, 20-closingTimeTent,
% 21-reopeningTimeTent, 22-peakTimeBlink, 23-peakTimeTent,
% 24-peakMaxBlink, 25-peakMaxTent

n = EEG.trials;
m = size(EEG.data,2);
time = zeros(n,1);  blink_freq = [];    durationMean = [];  durationStd = [];
duration_id = 1:5;  closing_id = [18 20];  reopening_id = [19 21];
timeShut_id = 15:17;    AVRB_id = 9:14;

bid = [];
for idx = 1:n
    time(idx) = (idx-1/2) * m / (EEG.srate);
    t_start = (1+(idx-1)*m)/EEG.srate;
    t_end = idx*m/EEG.srate;
    blink_id = find(blinkProp(:,22) > t_start & blinkProp(:,22) < t_end);
    tmp_prop = blinkProp(blink_id,:);
    mean_vals = mean(tmp_prop,1);     % Returns a row vector of the mean values of the diff properties
    std_vals = std(tmp_prop,0,1);
    
    if isempty(blink_id)
        bid = [bid idx];
    end
    
    blink_freq(idx,1) = length(blink_id);
    durationMean(idx,:) = mean_vals(1,duration_id);
    durationStd(idx,:) = std_vals(1,duration_id);
    closingTimeMean(idx,:) = mean_vals(1,closing_id);
    closingTimeStd(idx,:) = std_vals(1,closing_id);
    reopeningTimeMean(idx,:) = mean_vals(1,reopening_id);
    reopeningTimeStd(idx,:) = std_vals(1,reopening_id);
    timeShutMean(idx,:) = mean_vals(1,timeShut_id);
    timeShutStd(idx,:) = std_vals(1,timeShut_id);
    AVRB_Mean(idx,:) = mean_vals(1,AVRB_id);
    AVRB_Std(idx,:) = std_vals(1,AVRB_id);
end

blink_freq = blink_freq * (60/epochLen);        % Blink frequency (in mins)
% Call function to find reaction times and subjective drowsiness scores,
% along with their latencies

id = unique([id' bid]);
% Remove epochs which have no local mean reaction time (No gold standard!)
blink_freq(id) = [];    durationMean(id,:) = [];    durationStd(id,:) = [];
closingTimeMean(id,:) = [];  closingTimeStd(id,:) = [];  reopeningTimeMean(id,:) = [];
reopeningTimeStd(id,:) = [];  AVRB_Mean(id,:) = [];   AVRB_Std(id,:) = [];
RT_lm(id) = [];
time(id) = [];

% Create feature array for blink class data
blink_CD = [];

% disp('Blink frequency correlation coefficients')
[RHO,PVAL] = corr(RT_lm,blink_freq,'Type','Spearman');
if PVAL < signf_thresh
    blink_CD = [blink_CD blink_freq];
end

PVALtmp1 = 1;       PVALtmp2 = 1;       tmp_CD1=[];     tmp_CD2=[];
% disp('blink duration correlations')
for idx = 1:length(duration_id)
%   disp([num2str(idx),' mean correlation coefficients'])
    [RHO,PVAL] = corr(RT_lm(:,1),durationMean(:,idx),'Type','Spearman');
    if PVAL < signf_thresh & PVAL < PVALtmp1
    PVALtmp1 = PVAL;
    tmp_CD1 = durationMean(:,idx);
    end
    
%   disp([num2str(idx),' std correlation coefficients'])
    [RHO,PVAL] = corr(RT_lm(:,1),durationStd(:,idx),'Type','Spearman');
    if PVAL < signf_thresh & PVAL < PVALtmp2
    PVALtmp2 = PVAL;
    tmp_CD2 = durationStd(:,idx);
    end
end
blink_CD = [blink_CD tmp_CD1 tmp_CD2];


PVALtmp1 = 1;       PVALtmp2 = 1;       tmp_CD1=[];     tmp_CD2=[];
% disp('closing time correlations')
for idx = 1:length(closing_id)
%     disp([num2str(idx),' mean correlation coefficients'])
    [RHO,PVAL] = corr(RT_lm(:,1),closingTimeMean(:,idx),'Type','Spearman');
    if PVAL < signf_thresh & PVAL < PVALtmp1
        PVALtmp1 = PVAL;
        tmp_CD1 = closingTimeMean(:,idx);
    end
    
%     disp([num2str(idx),' std correlation coefficients'])
    [RHO,PVAL] = corr(RT_lm(:,1),closingTimeStd(:,idx),'Type','Spearman');
    if PVAL < signf_thresh & PVAL < PVALtmp2
        PVALtmp2 = PVAL;
        tmp_CD2 = closingTimeStd(:,idx);
    end
end
blink_CD = [blink_CD tmp_CD1 tmp_CD2];


PVALtmp1 = 1;       PVALtmp2 = 1;       tmp_CD1=[];     tmp_CD2=[];
% disp('reopening time correlations')
for idx = 1:length(reopening_id)
%     disp([num2str(idx),' mean correlation coefficients'])
    [RHO,PVAL] = corr(RT_lm(:,1),reopeningTimeMean(:,idx),'Type','Spearman');
    if PVAL < signf_thresh & PVAL < PVALtmp1
        PVALtmp1 = PVAL;
        tmp_CD1 = reopeningTimeMean(:,idx);
    end
    
%     disp([num2str(idx),' std correlation coefficients'])
    [RHO,PVAL] = corr(RT_lm(:,1),reopeningTimeStd(:,idx),'Type','Spearman');
    if PVAL < signf_thresh & PVAL < PVALtmp2
        PVALtmp2 = PVAL;
        tmp_CD2 = reopeningTimeStd(:,idx);
    end
end
blink_CD = [blink_CD tmp_CD1 tmp_CD2];


PVALtmp1 = 1;       PVALtmp2 = 1;       tmp_CD1=[];     tmp_CD2=[];
% disp('AVRB correlations')
for idx = 1:length(AVRB_id)
%     disp([num2str(idx),' mean correlation coefficients'])
    [RHO,PVAL] = corr(RT_lm(:,1),AVRB_Mean(:,idx),'Type','Spearman');
    if PVAL < signf_thresh & PVAL < PVALtmp1
        PVALtmp1 = PVAL;
        tmp_CD1 = AVRB_Mean(:,idx);
    end
    
%     disp([num2str(idx),' std correlation coefficients'])
    [RHO,PVAL] = corr(RT_lm(:,1),AVRB_Std(:,idx),'Type','Spearman');
    if PVAL < signf_thresh & PVAL < PVALtmp2
        PVALtmp2 = PVAL;
        tmp_CD2 = AVRB_Std(:,idx);
    end
end

blink_CD = [blink_CD tmp_CD1 tmp_CD2];
end

