function [fitresult, gof, DPF, DPP] = PeakFit(fRangefit, PowerProfile,Ep,fr_bef,fr_aft,time,RT_lm)
%PeakFit(fRangefit, PowerProfile,Ep,fr_bef,fr_aft)
%  Author: Tom Matthews, Part III Cambridge Physics student
%  This function performs a peak fitting on the power spectral density for
%  a given epoch and identifies the dominant peak. It then outputs the
%  dominant peak power and dominant peak frequency.
% 
%  Data for alpha peak fit:
%      X Input : fRange
%      Y Output: alphaProfile
%  Output:
%      fitresult : a fit object representing the fit.
%      gof : structure with goodness-of fit info.
%      DPF : Dominant peak frequency
%      DPP : Dominant peak power

%% Fit: 'Identify and characterise local maxima'.
[xData, yData] = prepareCurveData(fRangefit, PowerProfile);

% Initialise parameters
plotfit = false;
paramWarning = false;
frqLwr = xData(1) + fr_bef;
frqUpr = xData(end) - fr_aft;

% Identify potential peaks
xDataTrunc = xData(xData>=frqLwr & xData<=frqUpr);
yDataTrunc = yData(xData>=frqLwr & xData<=frqUpr);
MPprom = 0.15;
MPdist = 0.35;
[pks,locs,w] = findpeaks(yDataTrunc,xDataTrunc,'MinPeakProminence',MPprom,'MinPeakDistance',...
                MPdist,'Annotate','extents','SortStr','descend','WidthReference','halfheight');

% Iteratively relax conditions for indentifying peaks until at least 4
% peaks are found between 8 and 13 Hz.
Npeaks = 4;
count = 0;
decrement = 0.01;
while length(pks) < Npeaks
    count = count + 1;
    MPprom = MPprom - decrement;
    MPdist = MPdist - decrement;
    [pks,locs,w] = findpeaks(yDataTrunc,xDataTrunc,'MinPeakProminence',MPprom,'MinPeakDistance',...
                    MPdist,'Annotate','extents','SortStr','descend','WidthReference','halfheight');
                   
%     disp(['PEAK NUMBER: ' num2str(length(pks))])
%     if MPprom < 0.1
%         paramWarning = true;
%         disp('Warning: peak selection criterion severely relaxed')
%         disp(num2str(['MPprom = ' num2str(MPprom) ', MPdist = ' num2str(MPdist)]))
%     end
    
    if count >= 10 & ~isempty(length(pks))
        Npeaks = length(pks);
        break
    elseif count >= 14 & ~isempty(length(pks))
        decrement = 0.5 * decrement;
    end
end

if paramWarning
    disp(num2str(['MPprom = ' num2str(MPprom) ', MPdist = ' num2str(MPdist)]))
end

% Initialise fitting parameters, initialised such that Gaussians will only
% be fit to peaks successfully identified in the prescribed freq band.
Gpks(1:4) = 1;
Glocs(1:4) = 25;
mrgn(1:4) = 10;
std(1:4) = 1;
stdUB(1:4) = 1000;
ampUB(1:4) = 0;

% Npeaks largest Gaussian peaks and their locations, along with the lower and
% upper bound for their means
Gpks(1:Npeaks) = pks(1:Npeaks);
Glocs(1:Npeaks) = locs(1:Npeaks);
mrgn(1:Npeaks) = 0.25;        % Constrain the Gaussian location around peaks present
std(1:Npeaks) = sqrt(2)/2.36 .* w(1:Npeaks);      % Convert FWHM to stdev
stdUB(1:Npeaks) = 0.75;
ampUB(1:Npeaks) = Inf;
GlocsLB = Glocs - mrgn;
GlocsUB = Glocs + mrgn;

% Set up fittype and options.
% Use superposition of 1/f transient and multiple Gaussian to fit to peaks.
expr = ['a0+b0/x+c0*exp(-((x-d0)/e0)^2)+c1*exp(-((x-d1)/e1)^2)+c2*exp(-((x-d2)/e2)^2)'...
            '+c3*exp(-((x-d3)/e3)^2)'];

ft = fittype(expr,'independent','x','dependent','y');
opts = fitoptions('Method','NonlinearLeastSquares');
opts.Display = 'Off';
% Gauss properties[ ~ , ~  ,  amp   ,  mean  ,  stdev ]
opts.Lower =      [-Inf 0, 0 0 0 0, GlocsLB(1) GlocsLB(2) GlocsLB(3) GlocsLB(4), 0 0 0 0];
opts.StartPoint = [0 1, Gpks(1) Gpks(2) Gpks(3) Gpks(4),...
                        Glocs(1) Glocs(2) Glocs(3) Glocs(4), std(1) std(2) std(3) std(4)];
opts.Upper = [Inf Inf, ampUB(1) ampUB(2) ampUB(3) ampUB(4), GlocsUB(1) GlocsUB(2)...
                        GlocsUB(3) GlocsUB(4) ,stdUB(1) stdUB(2) stdUB(3) stdUB(4)];

% Weight the model to prioritise a good fit around the peaks.
wt_minim = 0.5;
wts = ones(length(xData),1);
wts(xData < (frqLwr-0.5)) = wt_minim * wts(xData < (frqLwr-0.5));
wts(xData > (frqUpr+0.5)) = wt_minim * wts(xData > (frqUpr+0.5));
opts.Weights = wts;

% Fit model to data.
[fitresult, gof] = fit(xData, yData, ft, opts);

%% Select the dominant peak and save its parameters.

amplitudes = [fitresult.c0 fitresult.c1 fitresult.c2 fitresult.c3];
frequencies = [fitresult.d0 fitresult.d1 fitresult.d2 fitresult.d3];
stds = [fitresult.e0 fitresult.e1 fitresult.e2 fitresult.e3];
powerInpeak = amplitudes .* stds;
[~, dom_id] = max(powerInpeak);

% Extract dominant peak frequency and dominant peak power
DPF = frequencies(dom_id);
DPP = powerInpeak(dom_id);

if plotfit
    % Create a figure for the plots.
    figure('Name', ['Peak fit for epoch ' num2str(Ep) ' at time ' num2str(time(Ep)) ' s, reaction time = ' num2str(RT_lm(Ep)) ' s']);

    % Plot fit with data.
    subplot( 2, 1, 1 );
    h = plot( fitresult, xData, yData );
    legend( h, 'Data', 'Model fit', 'Location', 'NorthEast' );
    % Label axes
    xlabel('Frequency / Hz')
    ylabel('Power')
    grid on

    % Plot residuals.
    subplot( 2, 1, 2 );
    h = plot( fitresult, xData, yData, 'residuals' );
    legend( h, 'alpha peak fit - residuals', 'Zero Line', 'Location', 'NorthEast' );
    % Label axes
    xlabel fRange
    ylabel Power
    grid on
end
