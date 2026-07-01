% model2_extended_comparison.m
%
% Extends the regression comparison with Deming, Theil-Sen, and weighted
% OLS alongside York (buggy/corrected) and MA.

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
Y_rel = s_raw./m_raw;
lm    = log(m_raw);
slm   = abs(sm_raw./m_raw);

t2u_b = abs(Y_rel./m_raw).*sqrt((ss_raw./Y_rel).^2+(sm_raw./m_raw).^2);
t2u_c = abs(Y_rel).*sqrt((ss_raw./s_raw).^2+(sm_raw./m_raw).^2);

lat_v = repmat((89:-2:-69)',1,180);
w = lat_v; w(isnan(lm))=NaN;
w = cosd(w); w = w.*sum(~isnan(w(:)))./nansum(w(:));

%% ── York regressions ─────────────────────────────────────────────────────────
t2u_b(t2u_b==0)=NaN;
vb = ~isnan(t2u_b(:))&~isnan(lm(:))&~isnan(slm(:))&~isnan(w(:));
[INT_yb,SLP_yb,~,SLPu_yb] = york_reg(lm(vb)',Y_rel(vb)',...
    slm(vb)'./sqrt(w(vb)'), t2u_b(vb)'./sqrt(w(vb)'));

t2u_c(t2u_c==0)=NaN;
vc = ~isnan(t2u_c(:))&~isnan(lm(:))&~isnan(slm(:))&~isnan(w(:));
[INT_yc,SLP_yc,~,SLPu_yc] = york_reg(lm(vc)',Y_rel(vc)',...
    slm(vc)'./sqrt(w(vc)'), t2u_c(vc)'./sqrt(w(vc)'));

%% ── Common data for non-York methods ─────────────────────────────────────────
valid = vc;
X = lm(valid); Y = Y_rel(valid); W = w(valid);
Wn = W ./ sum(W);
Nc = sum(valid);

xbar = sum(Wn.*X);
ybar = sum(Wn.*Y);
Sxx  = sum(Wn.*(X-xbar).^2);
Syy  = sum(Wn.*(Y-ybar).^2);
Sxy  = sum(Wn.*(X-xbar).*(Y-ybar));

%% ── Weighted OLS ─────────────────────────────────────────────────────────────
b_OLS = Sxy / Sxx;
a_OLS = ybar - b_OLS*xbar;

%% ── Major Axis (MA) ─────────────────────────────────────────────────────────
b_MA = (Syy - Sxx + sqrt((Syy-Sxx)^2 + 4*Sxy^2)) / (2*Sxy);
a_MA = ybar - b_MA*xbar;

%% ── Deming regression ────────────────────────────────────────────────────────
% Estimate variance ratio lambda from mean uncertainties (corrected)
uY_c = t2u_c(vc); uX_c = slm(vc);
lambda = nanmean(uY_c.^2) / nanmean(uX_c.^2);
b_Dem = (Syy - lambda*Sxx + sqrt((Syy - lambda*Sxx)^2 + 4*lambda*Sxy^2)) / (2*Sxy);
a_Dem = ybar - b_Dem*xbar;
fprintf('Deming variance ratio lambda = %.3f\n', lambda);

%% ── Theil-Sen (non-parametric) ───────────────────────────────────────────────
% Subsample for computational feasibility (~27M pairs at N=7400)
fprintf('Computing Theil-Sen (subsampled) ...\n');
rng(42);
Nsub = min(5000, Nc);
idx_sub = randperm(Nc, Nsub);
Xs = X(idx_sub); Ys = Y(idx_sub);

Npairs = Nsub*(Nsub-1)/2;
slopes_ts = nan(Npairs, 1);
k = 0;
for i = 1:Nsub-1
    dx = Xs(i+1:end) - Xs(i);
    dy = Ys(i+1:end) - Ys(i);
    nz = dx ~= 0;
    nn = sum(nz);
    slopes_ts(k+1:k+nn) = dy(nz) ./ dx(nz);
    k = k + nn;
end
slopes_ts = slopes_ts(1:k);
b_TS = median(slopes_ts);
a_TS = median(Y) - b_TS * median(X);
fprintf('  %d pairwise slopes computed (from %d subsampled points)\n', k, Nsub);

%% ── Analytical standard errors ────────────────────────────────────────────────
fprintf('Computing analytical standard errors ...\n');

% Weighted OLS: SE = sqrt(MSE / sum(w*(x-xbar)^2))
resid_ols = Y - a_OLS - b_OLS*X;
MSE_ols   = sum(W.*resid_ols.^2) / (Nc - 2);
se_OLS    = sqrt(MSE_ols / sum(W.*(X-xbar).^2));

% MA: eigenvalue-based formula
% λ₁, λ₂ = eigenvalues of the weighted covariance matrix
D    = sqrt((Sxx-Syy)^2 + 4*Sxy^2);
lam1 = (Sxx + Syy + D) / 2;
lam2 = (Sxx + Syy - D) / 2;
se_MA = sqrt(lam2/lam1 * (1 + b_MA^2)^2 / (Nc - 2));

% Deming: generalisation of MA formula with variance ratio lambda
resid_dem = Y - a_Dem - b_Dem*X;
MSE_dem   = sum(W.*resid_dem.^2) / (Nc - 2);
se_Dem    = sqrt(MSE_dem * (1 + b_Dem^2) / sum(W.*(X-xbar).^2));

% Theil-Sen: SE from the interquartile range of pairwise slopes
% SE ≈ (Q3 - Q1) / (2 * 0.6745 * sqrt(N))  (normal approximation)
Q1_ts = quantile(slopes_ts, 0.25);
Q3_ts = quantile(slopes_ts, 0.75);
se_TS = (Q3_ts - Q1_ts) / (2 * 0.6745 * sqrt(Nc));

%% ── Convert and display ──────────────────────────────────────────────────────
conv = log(10) * 1000;

methods = {
    'York (buggy)',     SLP_yb*conv, SLPu_yb*conv, INT_yb, SLP_yb;
    'York (corrected)', SLP_yc*conv, SLPu_yc*conv, INT_yc, SLP_yc;
    'Weighted OLS',     b_OLS*conv,  se_OLS*conv,   a_OLS,  b_OLS;
    'Major Axis (MA)',  b_MA*conv,   se_MA*conv,    a_MA,   b_MA;
    'Deming',           b_Dem*conv,  se_Dem*conv,   a_Dem,  b_Dem;
    'Theil-Sen',        b_TS*conv,   se_TS*conv,    a_TS,   b_TS;
};

fprintf('\n%s\n', repmat('=',1,55));
fprintf('%-22s  %s\n','Method','Slope (%/dec per 10-fold)');
fprintf('%s\n', repmat('-',1,55));
for k=1:size(methods,1)
    fprintf('%-22s  %+.2f +/- %.2f\n', methods{k,1}, methods{k,2}, methods{k,3});
end
fprintf('%-22s  %+.2f\n','Cael published', 1.20);
fprintf('%s\n', repmat('=',1,55));

%% ── Figure ───────────────────────────────────────────────────────────────────
rng(42);
fig = figure('Position',[50 50 700 540],'Color','w');
ax  = axes(fig);

Np = sum(valid); pidx = randperm(Np, min(8000,Np));
scatter(ax, X(pidx)/log(10), Y(pidx)*1000, 10, [0.75 0.85 0.93], 'o', ...
        'filled', 'MarkerFaceAlpha', 0.3);
hold(ax,'on');

xfit = linspace(min(X), max(X), 300);

colors = {
    [0.70 0.70 0.70];   % York buggy — light grey
    [0    0    0   ];    % York corrected — black
    [0.55 0.55 0.55];   % Weighted OLS — dark grey
    [0.85 0.33 0.10];   % MA — orange
    [0.00 0.45 0.74];   % Deming — blue
    [0.47 0.67 0.19];   % Theil-Sen — green
};
lw = [1.5, 2.2, 1.4, 1.8, 1.8, 1.8];
ls = {'--', '-', ':', '-', '-', '-'};

handles = gobjects(size(methods,1),1);
for k = 1:size(methods,1)
    INT_k = methods{k,4}; SLP_k = methods{k,5};
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

leg_labels = cell(size(methods,1),1);
for k=1:size(methods,1)
    leg_labels{k} = sprintf('%s: %+.2f \\pm %.2f', methods{k,1}, methods{k,2}, methods{k,3});
end
legend(ax, handles, leg_labels, 'FontSize', 8.5, 'Location', 'northwest', 'Box', 'on');

out = fullfile(OUT_DIR, 'fig_model2_extended.png');
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
