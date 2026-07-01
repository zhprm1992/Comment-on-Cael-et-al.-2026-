% model2_regression_comparison.m
%
% Tests three Model II regression methods (MA, SMA, RMA) alongside
% York regression (buggy and corrected) to assess robustness of the
% correction effect.
%
% Key point: MA/SMA/RMA do not use uncertainty estimates as weights, so
% they give a single slope (same for buggy and corrected data).
% Comparing York (buggy), York (corrected), and the three unweighted
% methods shows whether the corrected York slope is consistent with the
% underlying data relationship.

clear; close all;
OUT_DIR = 'd:\Prj3\Cael_2026';

%% ── Load cached data and reproduce pipeline ──────────────────────────────────
fprintf('Loading cached data ...\n');
tmp = load(fullfile(OUT_DIR,'C1_mo_9km.mat')); C1_mo = tmp.C1_mo; clear tmp;
tmp = load(fullfile(OUT_DIR,'COV_mo_9km.mat')); COV_mo = tmp.COV_mo; clear tmp;

P = 0.70; NYR = 20;
C1_th = C1_mo; C1_th(COV_mo < P) = NaN;

C2 = nan(90,180,NYR); C4 = nan(45,90,NYR);
for i = 1:NYR
    mo = (i-1)*12+1:12*i;
    C1a = mean(C1_th(:,:,mo), 3, 'omitnan');
    d = (C1a(1:2:end,:)+C1a(2:2:end,:))/2;
    d = (d(:,1:2:end)+d(:,2:2:end))/2;
    C2(:,:,i) = d;
    d2 = (d(1:2:end,:)+d(2:2:end,:))/2;
    d2 = (d2(:,1:2:end)+d2(:,2:2:end))/2;
    C4(:,:,i) = d2;
end

N = sum(~isnan(C4),3); N(N<20)=0; N(N==20)=1;
N(40,12)=0; N(4:8,23:29)=0; N(13,79)=0; N(22,76)=0;
N(13:14,47:50)=0; N(4:5,52:61)=0;
try
    N2 = imresize(double(N),2); N2(N2>0.5)=1; N2(N2<=0.5)=0;
catch
    N2 = repelem(double(N),2,2);
end
N2 = logical(N2);
for i=1:NYR; d=C2(:,:,i); d(~N2)=NaN; C2(:,:,i)=d; end

fprintf('Running OLS per cell ...\n');
t_vec = (-9.5:9.5)';
t2=zeros(90,180); m2=zeros(90,180); t2u=zeros(90,180); m2u=zeros(90,180);
for i=1:90
    for j=1:180
        if N2(i,j)
            y=squeeze(C2(i,j,:));
            if sum(~isnan(y))==NYR
                mdl=fitlm(t_vec,y);
                t2(i,j)=mdl.Coefficients{2,1}; m2(i,j)=mdl.Coefficients{1,1};
                t2u(i,j)=mdl.Coefficients{2,2}; m2u(i,j)=mdl.Coefficients{1,2};
            end
        end
    end
end
N2_80 = N2(1:80,:);
t2=t2(1:80,:); m2=m2(1:80,:); t2u=t2u(1:80,:); m2u=m2u(1:80,:);
t2(~N2_80)=NaN; m2(~N2_80)=NaN; t2u(~N2_80)=NaN; m2u(~N2_80)=NaN;

%% ── Prepare regression variables ─────────────────────────────────────────────
s_raw=t2; ss_raw=t2u; m_raw=m2; sm_raw=m2u;
Y_rel = s_raw./m_raw;          % relative trend (identical for all methods)
lm    = log(m_raw);            % log mean Chl (X)
slm   = abs(sm_raw./m_raw);    % relative uncertainty of mean (X uncertainty)

% Buggy and corrected Y uncertainties (used only by York)
t2u_b = abs(Y_rel./m_raw).*sqrt((ss_raw./Y_rel).^2+(sm_raw./m_raw).^2);
t2u_c = abs(Y_rel).*sqrt((ss_raw./s_raw).^2+(sm_raw./m_raw).^2);

% Cosine-latitude area weights
lat_v = repmat((89:-2:-69)',1,180);
w = lat_v; w(isnan(lm))=NaN;
w = cosd(w); w = w.*sum(~isnan(w(:)))./nansum(w(:));

%% ── York regressions ─────────────────────────────────────────────────────────
t2u_b(t2u_b==0)=NaN;
vb = ~isnan(t2u_b(:))&~isnan(lm(:))&~isnan(slm(:))&~isnan(w(:));
[INT_b,SLP_b,~,SLPu_b] = york_reg(lm(vb)',Y_rel(vb)',...
    slm(vb)'./sqrt(w(vb)'), t2u_b(vb)'./sqrt(w(vb)'));

t2u_c(t2u_c==0)=NaN;
vc = ~isnan(t2u_c(:))&~isnan(lm(:))&~isnan(slm(:))&~isnan(w(:));
[INT_c,SLP_c,~,SLPu_c] = york_reg(lm(vc)',Y_rel(vc)',...
    slm(vc)'./sqrt(w(vc)'), t2u_c(vc)'./sqrt(w(vc)'));

%% ── Model II regressions (area-weighted, no uncertainty estimates) ───────────
% Use the corrected valid set for X, Y (same data, no uncertainty weights)
valid = vc;
X = lm(valid); Y = Y_rel(valid); W = w(valid);
W = W ./ sum(W);   % normalise weights to sum to 1

xbar = sum(W.*X);
ybar = sum(W.*Y);
Sxx  = sum(W.*(X-xbar).^2);
Syy  = sum(W.*(Y-ybar).^2);
Sxy  = sum(W.*(X-xbar).*(Y-ybar));
r    = Sxy / sqrt(Sxx*Syy);

% MA (Major Axis): minimises sum of squared orthogonal distances
b_MA = (Syy - Sxx + sqrt((Syy-Sxx)^2 + 4*Sxy^2)) / (2*Sxy);
a_MA = ybar - b_MA*xbar;

% SMA (Standardised Major Axis): slope = sign(r) * SD_Y / SD_X
b_SMA = sign(Sxy) * sqrt(Syy/Sxx);
a_SMA = ybar - b_SMA*xbar;

% RMA (Reduced Major Axis / geometric mean): geometric mean of the two OLS slopes
b_OLS_YX = Sxy/Sxx;           % OLS slope of Y on X
b_OLS_XY = Sxy/Syy;           % OLS slope of X on Y (inverted: Sxy/Syy)
b_RMA    = sign(Sxy) * sqrt(b_OLS_YX / b_OLS_XY);  % = sign(r)*sqrt(Syy/Sxx) = SMA
a_RMA    = ybar - b_RMA*xbar;

%% ── Convert slopes to %/decade per 10-fold ───────────────────────────────────
conv = log(10) * 1000;
results = {
    'York (buggy)',        SLP_b*conv,  SLPu_b*conv, INT_b,  SLP_b;
    'York (corrected)',    SLP_c*conv,  SLPu_c*conv, INT_c,  SLP_c;
    'Major Axis (MA)',     b_MA*conv,   NaN,          a_MA,   b_MA;
    'Standardized MA (SMA)', b_SMA*conv, NaN,         a_SMA,  b_SMA;
    'Reduced MA (RMA)',    b_RMA*conv,  NaN,          a_RMA,  b_RMA;
};

fprintf('\n%s\n', repmat('=',1,55));
fprintf('%-28s  %s\n','Method','Slope (%/dec per 10-fold)');
fprintf('%s\n', repmat('-',1,55));
for k=1:size(results,1)
    if ~isnan(results{k,3})
        fprintf('%-28s  %+.2f +/- %.2f\n', results{k,1}, results{k,2}, results{k,3});
    else
        fprintf('%-28s  %+.2f\n', results{k,1}, results{k,2});
    end
end
fprintf('%-28s  %+.2f\n','Cael published', 1.20);
fprintf('%s\n', repmat('=',1,55));
fprintf('\nWeighted correlation r = %.3f  (N = %d cells)\n', r, sum(valid));

%% ── Figure: all regression lines on one scatter ──────────────────────────────
rng(42);
fig = figure('Position',[50 50 680 530],'Color','w');
ax  = axes(fig);

% Scatter (same data for all methods)
Np = sum(valid); idx = randperm(Np, min(8000,Np));
Xp = X(idx); Yp = Y(idx);
scatter(ax, Xp/log(10), Yp*1000, 10, [0.75 0.85 0.93], 'o', ...
        'filled', 'MarkerFaceAlpha', 0.3);
hold(ax,'on');

% Regression lines
xfit = linspace(min(X), max(X), 300);
colors = {[0.6 0.6 0.6],    ...  % York buggy — grey
          [0   0   0  ],    ...  % York corrected — black
          [0.85 0.33 0.10], ...  % MA — orange
          [0   0.45 0.74],  ...  % SMA — blue
          [0.47 0.67 0.19]};     % RMA — green

lw = [1.5, 2.2, 1.8, 1.8, 1.8];
ls = {'--','-','-','-','-'};

handles = gobjects(5,1);
for k = 1:5
    INT_k = results{k,4}; SLP_k = results{k,5};
    handles(k) = plot(ax, xfit/log(10), (INT_k + SLP_k*xfit)*1000, ...
        'Color', colors{k}, 'LineWidth', lw(k), 'LineStyle', ls{k});
end

yline(ax,0,'Color',[.6 .6 .6],'LineWidth',.6,'LineStyle',':');
xline(ax,0,'Color',[.6 .6 .6],'LineWidth',.6,'LineStyle',':');

xlim(ax,[-2.0 0.7]); ylim(ax,[-65 65]);
set(ax,'XTick',-2:0.5:0.5,'FontSize',10,'Box','on');
xlabel(ax,'log_{10}(mean Chl) [mg m^{-3}]','FontSize',11);
ylabel(ax,'Chl trend [% decade^{-1}]','FontSize',11);
ax.Toolbar.Visible = 'off';

% Legend with slope values
leg_labels = cell(5,1);
for k=1:5
    if ~isnan(results{k,3})
        leg_labels{k} = sprintf('%s: %+.2f \\pm %.2f', results{k,1}, results{k,2}, results{k,3});
    else
        leg_labels{k} = sprintf('%s: %+.2f', results{k,1}, results{k,2});
    end
end
legend(ax, handles, leg_labels, 'FontSize', 9, 'Location', 'northwest', 'Box', 'on');

title(ax, 'Model II regression comparison', ...
      'FontSize', 10, 'FontWeight', 'normal');

out = fullfile(OUT_DIR, 'fig_model2_comparison.png');
exportgraphics(fig, out, 'Resolution', 200);
fprintf('\nFigure saved: %s\n', out);

%% ── Local function ───────────────────────────────────────────────────────────
function [a,b,sa,sb] = york_reg(X,Y,sX,sY)
X=X(:); Y=Y(:); sX=sX(:); sY=sY(:);
wX=1./sX.^2; wY=1./sY.^2;
p=polyfit(X,Y,1); b=p(1);
for iter=1:200
    W=wX.*wY./(wX+b^2.*wY);
    Xb=sum(W.*X)/sum(W); Yb=sum(W.*Y)/sum(W);
    U=X-Xb; V=Y-Yb;
    beta=W.*(U./wY+b.*V./wX);
    den=sum(W.*beta.*U); if den==0, break; end
    bn=sum(W.*beta.*V)/den;
    if abs(bn-b)<1e-12, b=bn; break; end
    b=bn;
end
a=Yb-b*Xb; x=Xb+beta; xb=sum(W.*x)/sum(W); u=x-xb;
sb=sqrt(1/sum(W.*u.^2)); sa=sqrt(1/sum(W)+xb^2*sb^2);
end
