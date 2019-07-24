function [SpectralPower, SpectralCentroid, SpectralVariance, SE, DPF, DPP] ...
            = classify_computePowerfts(EEG,id,epochLen,time,RT_lm)
%classify_computePowerfts:

% Author: Tom Matthews, Part III Cambridge Physics student
% Over multiple frequency bands, produces features associated with the
% variance of power (SpectralVariance), mean power frequency
% (SpectralCentroid). In each frequency band the most prominent local
% maxima is identified as the dominant peak, from which the dominant peak
% frequency (DPF) and dominant peak power within the peak (DPP) are
% extracted. The function also extracts the relative power in 2 Hz bins
% from 0.5 - 40 Hz and the relative power within the canonical freq bands
% (delta,theta,alpha,...).


nelec = EEG.nbchan;
ntrials = EEG.trials;

S=[];

%Allocating frequency bands..
startfreq = 0;
endfreq = 40;
countvar = 1;
for k = startfreq:2:endfreq-2
    
    fieldname = ['freqband' num2str(countvar)];
    S.(fieldname)(1) = k;
    S.(fieldname)(2) = k+2;
    countvar = countvar + 1;
   
end

S.freqband1(1) = 0.5; %Just changing the first freq band alone..
% %Temp code..

S.freqband21(1) = 0.5;
S.freqband21(2) = 4;
S.freqband22(1) = 4;
S.freqband22(2) = 8;
S.freqband23(1) = 8;
S.freqband23(2) = 13;
S.freqband24(1) = 13;
S.freqband24(2) = 20;
S.freqband25(1) = 13;
S.freqband25(2) = 40;

bincount = length(fieldnames(S));

%% Compute the centroid frequency and variance of each freq band..

fmin = 0.5;
fmax = 40;
Pow = [];

for k = 1:nelec
                
 % The function to have a spectrum for one electrode
 [ersp,itc,powbase,times,freqs,erspboot,itcboot,tfdata] = ...
             newtimef(EEG.data(k,:,:), EEG.pnts,[EEG.xmin EEG.xmax]*1000, EEG.srate, 0, ...
              'padratio', 2, 'freqs', [fmin fmax], ...
              'plotersp', 'off','plotitc','off','verbose','off');  
         
  Pow(:,:,:,k)  = tfdata.*conj(tfdata);
  
  %      So Pow is a 4D array with the first dimension corresponding to the
  %      frequencies, the second dimension corresponding to the times within
  %      an epoch for which we have the fourier transform, and the 3rd
  %      dimension corresponding to each epoch, and the 4th dimension
  %      being the electrodes. Therefore sum over the relevant frequencies
  %      in the first dimension and take the mean over the times within
  %      each epoch to get the average frequency decomposition of each epoch.
  
end
            
Fband = [];

for n = 1:bincount
    tagnamebegin = ['Band' num2str(n) '_' 'fBeg'];
    tagnameend = ['Band' num2str(n) '_' 'fEnd'];
    fieldname = ['freqband' num2str(n)];
    
    [~, Fband.(tagnamebegin)] = min(abs(freqs-S.(fieldname)(1)));
    [~, Fband.(tagnameend)]  =  min(abs(freqs-S.(fieldname)(2)));
end
  
% Compute power in a frequency band..
power = [];

% Compute total power across all frequency bands for each epoch
Powertmp = mean(squeeze(sum(Pow(1:size(Pow,1),:,:,:),1)),1);
Powertmp = squeeze(Powertmp);
totalPower = mean(Powertmp,2);

for n = 1:bincount
    fieldname = ['freqband' num2str(n)];
    tagnamebegin = ['Band' num2str(n) '_' 'fBeg'];
    tagnameend = ['Band' num2str(n) '_' 'fEnd'];
    
    % Partition the frequencies, then take the means over the times and
    % electrodes, store the final power spectrum density function in power.
    bandPowertmp = Pow(Fband.(tagnamebegin):Fband.(tagnameend),:,:,:);
    bandPowertmp = mean(bandPowertmp,4);
    bandPowertmp = squeeze(mean(bandPowertmp,2));
    power.(fieldname) = bandPowertmp ./ totalPower';
    % Now power.(fieldname) has epochs in columns and rows are the
    % normalised powers of the frequencies in the given frequency band
end

for n = 1:bincount
    fieldname = ['freqband' num2str(n)];
    tagnamebegin = ['Band' num2str(n) '_' 'fBeg'];
    tagnameend = ['Band' num2str(n) '_' 'fEnd'];
    freq_ids = (Fband.(tagnamebegin):Fband.(tagnameend))';
    fRange = freqs(freq_ids)';
    
    prob_norm = sum(power.(fieldname),1);
    F = (sum(power.(fieldname).*fRange,1) ./ prob_norm)';
    FF = (sum(power.(fieldname).*(fRange.^2),1) ./ prob_norm)';
    
    SpectralPower(:,n) = (sum(power.(fieldname),1))';
    
    % Only include the Spectral Variance for the classical frequency bands
    if n > 20
        SpectralVariance(:,n) = (FF - F.^2);
        SpectralCentroid(:,n) = F;
    end    
    % The rows corresponds to the epochs and the columns
    % to the frequency bands, with every band providing a feature
    % column.
end

SpectralVariance = SpectralVariance(:,any(SpectralVariance));
%% Compute Shannon entropy from information theory, measure of brain state complexity

bPowertmp = Pow(1:size(Pow,1),:,:,:);
bPowertmp = mean(bPowertmp,4);
bPowertmp = squeeze(mean(bPowertmp,2))./totalPower';
SE = -(sum(bPowertmp.*log2(bPowertmp),1))' / log2(size(bPowertmp,1));

%% Extract the dominant peak frequency and dominant peak power from each freq band..

% Next extend frequency range in each band 
delta = (fmax-fmin)/(size(freqs,2)-1);
fr_bef = 1;         id_bef = round(fr_bef/delta);
fr_aft = 3;         id_aft = round(fr_aft/delta);

EpRange = 1:ntrials;
bands = 23;
DPF = zeros(ntrials,length(bands));
DPP = zeros(ntrials,length(bands));
scale = 500;

% Compute the DPF and DPP features in 'bands', currently set to just the
% alpha band.
for Ep = EpRange
    cnt = 0;
    for k = bands
        cnt = cnt + 1;
        fieldname = ['freqband' num2str(k)];
        tagnamebegin = ['Band' num2str(k) '_' 'fBeg'];
        tagnameend = ['Band' num2str(k) '_' 'fEnd'];
        
        if k == 21
            fr_bef = 0;
            id_bef = 0;
        end
        
        if k == 25
           fr_aft = 0;
           id_aft = 0;
        end
        
        freq_idf = ((Fband.(tagnamebegin)-id_bef):(Fband.(tagnameend)+id_aft))';
        fRangefit = freqs(freq_idf)';
        
        power_tmp = squeeze(mean(Pow(freq_idf,:,Ep,:),4));  % Mean over electrodes
        PowerProfile = scale * squeeze(mean(power_tmp,2)) / totalPower(Ep);  % Mean over times and normalize
        
        [fitresult,gof,DPF(Ep,cnt),DPP(Ep,cnt)] = ...
            PeakFit(fRangefit,PowerProfile,Ep,fr_bef,fr_aft,time,RT_lm);
        
    end
end

% Remove any observations where there is no gold standard (RT's) available.
SpectralPower(id,:) = [];
SpectralCentroid(id,:) = [];
SpectralVariance(id,:) = [];
DPF(id,:) = [];
DPP(id,:) = [];
SE(id) = [];

end

