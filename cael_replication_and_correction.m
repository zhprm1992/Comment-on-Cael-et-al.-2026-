% cael_replication_and_correction.m
%
% Replicates Cael et al. (2026, GRL) Figure 2 and identifies the coding
% error in their uncertainty propagation.
%
% Pipeline:
%   MODIS-Aqua 9km monthly Chl -> 1° (coverage threshold P=0.70)
%   -> annual means -> 2° -> 4° (for mask) -> OLS per cell -> York regression
%
% Two York regressions are compared:
%   Buggy    : uncertainty formula from chl_ineq_github.m lines 149-150
%              (t2 overwritten to s/m, causing double-division by m)
%   Corrected: abs(t2_c) * sqrt((ss/s)^2 + (sm/m)^2)
%
% Output: fig2_panel1_comparison.png (two-panel scatter, buggy vs corrected)

clear; close all;
DATA_DIR = 'd:\Prj3\Cael_2026\modis_9km_l3m_monthly_chl';
OUT_DIR  = 'd:\Prj3\Cael_2026';
CACHE_C1  = fullfile(OUT_DIR, 'C1_mo_9km.mat');
CACHE_COV = fullfile(OUT_DIR, 'COV_mo_9km.mat');
NYR = 20;  NMO = 240;

%% ── Phase 1: Build monthly 1° Chl + coverage (cached) ───────────────────────
if exist(CACHE_C1,'file') && exist(CACHE_COV,'file')
    fprintf('Loading cached C1_mo and COV_mo ...\n');
    tmp = load(CACHE_C1);  C1_mo  = tmp.C1_mo;  clear tmp;
    tmp = load(CACHE_COV); COV_mo = tmp.COV_mo; clear tmp;
else
    d_list = dir(fullfile(DATA_DIR,'AQUA_MODIS.*.L3m.MO.CHL.chlor_a.9km.nc'));
    files  = sort({d_list.name})';
    assert(numel(files)==NMO);

    C1_mo  = nan(180, 360, NMO, 'single');
    COV_mo = nan(180, 360, NMO, 'single');   % fraction of valid 9km cells per 1°

    fprintf('Reading %d files ...\n', NMO);
    t0 = tic;
    for k = 1:NMO
        raw = single(ncread(fullfile(DATA_DIR,files{k}), 'chlor_a'));
        chl = raw';                            % (2160, 4320)
        chl(chl<=0 | chl>1000) = NaN;

        valid = single(~isnan(chl));           % 1 where valid

        % 9km -> 1° mean (nanmean via omitnan)
        tmp = reshape(chl,   [12,180,4320]);
        d1  = squeeze(mean(tmp,1,'omitnan'));
        tmp = reshape(d1,    [180,12,360]);
        C1_mo(:,:,k) = squeeze(mean(tmp,2,'omitnan'));

        % 9km -> 1° coverage fraction
        tmp = reshape(valid, [12,180,4320]);
        v1  = squeeze(mean(tmp,1));            % mean of 0/1 = fraction
        tmp = reshape(v1,    [180,12,360]);
        COV_mo(:,:,k) = squeeze(mean(tmp,2));

        if mod(k,12)==0
            fprintf('  Year %d  (%d/%d, %.0fs)\n', 2002+k/12, k, NMO, toc(t0));
        end
    end
    save(CACHE_C1,  'C1_mo',  '-v7.3');
    save(CACHE_COV, 'COV_mo', '-v7.3');
    fprintf('Cached to disk in %.0fs\n', toc(t0));
end

%% ── Phase 2: Sweep coverage threshold ───────────────────────────────────────
thresholds = [0.05 0.10 0.20 0.30 0.40 0.50 0.60 0.70 0.80 0.90];
cell_counts = nan(size(thresholds));

fprintf('\nSweeping coverage threshold (target N = 7376) ...\n');
fprintf('  %-10s  %s\n', 'Threshold', 'Open-ocean cells');

for ti = 1:numel(thresholds)
    P = thresholds(ti);

    % Apply threshold: NaN out 1° cells below coverage P
    C1_th = C1_mo;
    C1_th(COV_mo < P) = NaN;

    % Annual means at 1° then 2° then 4°
    C4 = nan(45,90,NYR);
    for i = 1:NYR
        mo = (i-1)*12+1:12*i;
        C1a = mean(C1_th(:,:,mo), 3, 'omitnan');   % 1° annual mean

        % 1° -> 2° (NaN-propagating)
        d = C1a;
        d = (d(1:2:end,:)+d(2:2:end,:))/2;
        d = (d(:,1:2:end)+d(:,2:2:end))/2;          % (90,180)

        % 2° -> 4° (NaN-propagating)
        d = (d(1:2:end,:)+d(2:2:end,:))/2;
        d = (d(:,1:2:end)+d(:,2:2:end))/2;          % (45,90)
        C4(:,:,i) = d;
    end

    % Cael mask at 4°
    N = sum(~isnan(C4),3);
    N(N<20)=0; N(N==20)=1;
    N(40,12)=0; N(4:8,23:29)=0; N(13,79)=0; N(22,76)=0;
    N(13:14,47:50)=0; N(4:5,52:61)=0;
    try
        N2 = imresize(double(N),2); N2(N2>0.5)=1; N2(N2<=0.5)=0;
    catch
        N2 = repelem(double(N),2,2);
    end
    cell_counts(ti) = sum(N2(:));
    fprintf('  P = %.2f        ->  %d cells\n', P, cell_counts(ti));
end

% Find threshold closest to 7376
[~, best_idx] = min(abs(cell_counts - 7376));
P_best = thresholds(best_idx);
fprintf('\nBest threshold: P = %.2f  (%d cells, target 7376)\n\n', ...
        P_best, cell_counts(best_idx));

%% ── Phase 3: Full regression with best threshold ─────────────────────────────
fprintf('Running full pipeline with P = %.2f ...\n', P_best);

C1_th = C1_mo;
C1_th(COV_mo < P_best) = NaN;

C2  = nan(90,180,NYR);
C4  = nan(45, 90,NYR);
for i = 1:NYR
    mo  = (i-1)*12+1:12*i;
    C1a = mean(C1_th(:,:,mo), 3, 'omitnan');
    d   = (C1a(1:2:end,:)+C1a(2:2:end,:))/2;
    d   = (d(:,1:2:end)+d(:,2:2:end))/2;
    C2(:,:,i) = d;
    d2  = (d(1:2:end,:)+d(2:2:end,:))/2;
    d2  = (d2(:,1:2:end)+d2(:,2:2:end))/2;
    C4(:,:,i) = d2;
end

% Mask
N = sum(~isnan(C4),3);
N(N<20)=0; N(N==20)=1;
N(40,12)=0; N(4:8,23:29)=0; N(13,79)=0; N(22,76)=0;
N(13:14,47:50)=0; N(4:5,52:61)=0;
try
    N2 = imresize(double(N),2); N2(N2>0.5)=1; N2(N2<=0.5)=0;
catch
    N2 = repelem(double(N),2,2);
end
N2 = logical(N2);
fprintf('  Open-ocean cells: %d  (Cael: ~7376)\n', sum(N2(:)));

% Apply mask & OLS
for i = 1:NYR
    d=C2(:,:,i); d(~N2)=NaN; C2(:,:,i)=d;
end
t_vec = (-9.5:9.5)';
t2=zeros(90,180); m2=zeros(90,180); t2u=zeros(90,180); m2u=zeros(90,180);
for i = 1:90
    for j = 1:180
        if N2(i,j)
            y = squeeze(C2(i,j,:));
            if sum(~isnan(y))==NYR
                mdl=fitlm(t_vec,y);
                t2(i,j)=mdl.Coefficients{2,1}; m2(i,j)=mdl.Coefficients{1,1};
                t2u(i,j)=mdl.Coefficients{2,2}; m2u(i,j)=mdl.Coefficients{1,2};
            end
        end
    end
    if mod(i,10)==0, fprintf('  OLS row %d/90\n',i); end
end

N2_80=N2(1:80,:);
t2=t2(1:80,:); m2=m2(1:80,:); t2u=t2u(1:80,:); m2u=m2u(1:80,:);
t2(~N2_80)=NaN; m2(~N2_80)=NaN; t2u(~N2_80)=NaN; m2u(~N2_80)=NaN;
fprintf('  Valid cells: %d\n', sum(~isnan(t2(:))));

% Uncertainty
s_raw=t2; ss_raw=t2u; m_raw=m2; sm_raw=m2u;

t2_b  = t2./m2;
t2u_b = abs(t2_b./m2).*sqrt((ss_raw./t2_b).^2+(m2u./m2).^2);  % buggy

t2_c  = s_raw./m_raw;
t2u_c = abs(t2_c).*sqrt((ss_raw./s_raw).^2+(sm_raw./m_raw).^2); % correct

lm  = log(m_raw);
slm = abs(sm_raw./m_raw);

% Latitude weights
lat_v = repmat((89:-2:-69)',1,180);
w = lat_v; w(isnan(lm))=NaN;
w = cosd(w); w = w.*sum(~isnan(w(:)))./nansum(w(:));

% York — buggy
t2u_b(t2u_b==0)=NaN;
vb=~isnan(t2u_b(:))&~isnan(lm(:))&~isnan(slm(:));
Wb=w(:); Wb=Wb(vb); Xb=lm(:); Xb=Xb(vb); Yb=t2_b(:); Yb=Yb(vb);
uXb=slm(:); uXb=uXb(vb); uYb=t2u_b(:); uYb=uYb(vb);
[INT_b,SLP_b,~,SLPu_b]=york_reg(Xb',Yb',uXb'./sqrt(Wb'),uYb'./sqrt(Wb'));
sl_b=SLP_b*log(10)*1000; se_b=SLPu_b*log(10)*1000;

% York — corrected
t2u_c(t2u_c==0)=NaN;
vc=~isnan(t2u_c(:))&~isnan(lm(:))&~isnan(slm(:));
Wc=w(:); Wc=Wc(vc); Xc=lm(:); Xc=Xc(vc); Yc=t2_c(:); Yc=Yc(vc);
uXc=slm(:); uXc=uXc(vc); uYc=t2u_c(:); uYc=uYc(vc);
[INT_c,SLP_c,~,SLPu_c]=york_reg(Xc',Yc',uXc'./sqrt(Wc'),uYc'./sqrt(Wc'));
sl_c=SLP_c*log(10)*1000; se_c=SLPu_c*log(10)*1000;

fprintf('\n%s\n', repmat('=',1,60));
fprintf('Coverage threshold : P = %.2f\n', P_best);
fprintf('Open-ocean cells   : %d  (Cael: ~7376)\n', sum(~isnan(t2(:))));
fprintf('Buggy  slope       : %+.2f +/- %.2f  %%/dec per 10-fold\n', sl_b, se_b);
fprintf('Correct slope      : %+.2f +/- %.2f  %%/dec per 10-fold\n', sl_c, se_c);
fprintf('Cael published     :  1.20        %%/dec per 10-fold\n');
fprintf('%s\n', repmat('=',1,60));

%% ── Figure ───────────────────────────────────────────────────────────────────
rng(42);
fig = figure('Position',[50 50 1100 470],'Color','w');
panels = {Xb,Yb,INT_b,SLP_b,sl_b,se_b,'(a)  Cael et al. (2026) — as published';
          Xc,Yc,INT_c,SLP_c,sl_c,se_c,'(b)  Corrected uncertainty propagation'};
for p=1:2
    ax=subplot(1,2,p);
    Xp=panels{p,1}; Yp=panels{p,2}; INTp=panels{p,3}; SLPp=panels{p,4};
    slp=panels{p,5}; sep=panels{p,6}; ttl=panels{p,7};
    Np=numel(Xp); idx=randperm(Np,min(8000,Np));
    hs=scatter(ax,Xp(idx)/log(10),Yp(idx)*1000,12,[0.20 0.45 0.70],'o','filled','MarkerFaceAlpha',0.22);
    hold(ax,'on');
    xfit=linspace(min(Xp),max(Xp),300);
    hl=plot(ax,xfit/log(10),(INTp+SLPp*xfit)*1000,'k-','LineWidth',2.2);
    yline(ax,0,'Color',[.55 .55 .55],'LineWidth',.7,'LineStyle',':');
    xline(ax,0,'Color',[.55 .55 .55],'LineWidth',.7,'LineStyle',':');
    xlim(ax,[-2.0 0.7]); ylim(ax,[-65 65]);
    set(ax,'XTick',-2:0.5:0.5);
    xlabel(ax,'log_{10}(mean Chl) [mg m^{-3}]','FontSize',11);
    if p==1, ylabel(ax,'Chl trend [% decade^{-1}]','FontSize',11); end
    title(ax,ttl,'FontSize',11,'FontWeight','normal');
    legend(ax,[hs,hl],{'2°\times2° grid cell', sprintf('%+.2f \\pm %.2f %%/dec per 10-fold',slp,sep)},...
           'FontSize',9,'Location','northwest','Box','on');
    set(ax,'FontSize',10,'Box','on');
    ax.Toolbar.Visible = 'off';
end
% sgtitle removed — moved to figure caption in document
out=fullfile(OUT_DIR,'fig2_panel1_comparison.png');
exportgraphics(fig,out,'Resolution',200);
fprintf('\nFigure saved: %s\n', out);

%% ── Local function ───────────────────────────────────────────────────────────
function [a,b,sa,sb]=york_reg(X,Y,sX,sY)
X=X(:); Y=Y(:); sX=sX(:); sY=sY(:);
wX=1./sX.^2; wY=1./sY.^2;
p=polyfit(X,Y,1); b=p(1);
for iter=1:200
    W=wX.*wY./(wX+b^2.*wY);
    Xb=sum(W.*X)/sum(W); Yb=sum(W.*Y)/sum(W);
    U=X-Xb; V=Y-Yb;
    beta=W.*(U./wY+b.*V./wX);
    den=sum(W.*beta.*U); if den==0,break;end
    bn=sum(W.*beta.*V)/den;
    if abs(bn-b)<1e-12,b=bn;break;end
    b=bn;
end
a=Yb-b*Xb; x=Xb+beta; xb=sum(W.*x)/sum(W); u=x-xb;
sb=sqrt(1/sum(W.*u.^2)); sa=sqrt(1/sum(W)+xb^2*sb^2);
end
