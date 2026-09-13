"""
Emit LaTeX table bodies for the appendix directly from results_v3.xlsx, so
no number is transcribed by hand.

    python -u gen_tables.py
"""
import pandas as pd
X = pd.ExcelFile('/mnt/user-data/uploads/results_v3.xlsx')
out = {}

def f(v, d=3, sign=True):
    if pd.isna(v): return '---'
    s = f'{v:+.{d}f}' if sign else f'{v:.{d}f}'
    return '$' + s.replace('-', '-') + '$'

# ---- C.1 robustness grid -------------------------------------------------
g = pd.read_excel(X, 'grid_v3')
rows = []
for _, r in g.iterrows():
    tgt = {'g_meandec_cpi':'', 'g_median_cpi':'', 'g_meandec_rpi_rb':'',
           'E_H_mean_dec_h10':' ($E^H$)', 'E_P_cpi':' ($E^P$)'}.get(r.target,'')
    code = r.label.split('_')[0] + tgt
    rows.append(' & '.join([
        code.replace('_','\\_'),
        f'{100*r.acceptance_rate:.2f}',
        f'{r.fevd_s_h0:.3f}',
        f(r.gap_h0), f'$[{r.gap_h0_p16:+.3f},\\,{r.gap_h0_p84:+.3f}]$',
        f(r.gap_h4), f'$[{r.gap_h4_p16:+.3f},\\,{r.gap_h4_p84:+.3f}]$',
    ]) + r' \\')
out['grid'] = '\n'.join(rows)

# ---- C.2 episodes --------------------------------------------------------
e = pd.read_excel(X, 'episodes_v3')
order = ['2008_financial_crisis','2011_rpi_peak','2015_quiet_control','2022_cost_of_living']
rows = []
for lab in order:
    r = e[e.label == lab].iloc[0]
    rows.append(' & '.join([
        lab.replace('_','\\_'), r.targets, f'{int(r.kept)}', f'{int(r.draws):,}',
        f'{100*r.acceptance_rate:.3f}', f'{r.fevd_s_h0:.3f}',
        f(r.gap_h0), f'$[{r.gap_h0_p16:+.3f},\\,{r.gap_h0_p84:+.3f}]$',
        f(r.gap_h4), f'$[{r.gap_h4_p16:+.3f},\\,{r.gap_h4_p84:+.3f}]$',
    ]) + r' \\')
out['episodes'] = '\n'.join(rows)

# ---- C.3 sensitivity -----------------------------------------------------
s = pd.read_excel(X, 'sensitivity_v3')
rows = []
for _, r in s.iterrows():
    nm = ('FEVD threshold $' + f'{r.value:.1f}$') if r.restriction=='fevd_threshold' \
         else ('Persistence through $h=' + f'{int(r.value)}$')
    rows.append(' & '.join([
        nm, f'{int(r.draws):,}', f'{100*r.acceptance_rate:.3f}', f'{r.fevd_s_h0:.3f}',
        f(r.gap_h4), f'$[{r.gap_h4_p16:+.3f},\\,{r.gap_h4_p84:+.3f}]$',
    ]) + r' \\')
out['sens'] = '\n'.join(rows)

# ---- C.4 full baseline IRF of the gap -----------------------------------
b = pd.read_excel(X, 'irf_baseline_v3')
gp = b[b.variable=='g_meandec_cpi'].sort_values('h')
bs = pd.read_excel(X, 'irf_bootstrap_v3').sort_values('h')
rows = []
for (_, i), (_, w) in zip(gp.iterrows(), bs.iterrows()):
    assert i.h == w.h
    rows.append(' & '.join([
        f'{int(i.h)}',
        f(i["median"]), f'$[{i.p16:+.3f},\\,{i.p84:+.3f}]$', ('yes' if i.excludes_zero else 'no'),
        f(w["median"]), f'$[{w.p16:+.3f},\\,{w.p84:+.3f}]$', ('yes' if w.excludes_zero else 'no'),
    ]) + r' \\')
out['irf'] = '\n'.join(rows)

# ---- C.5 FEVD ------------------------------------------------------------
fv = pd.read_excel(X, 'fevd_baseline_v3')
vars_ = ['q_t_commod_yoy','a_t_kilian','pi_t_rpi','s_t_asinh100','D_t_std_dec_h10','g_meandec_cpi']
rows = []
for h in [0,1,2,3,4,6,8,12,16,20]:
    cells = [f'{h}']
    for v in vars_:
        r = fv[(fv.h==h)&(fv.variable==v)].iloc[0]
        cells.append(f'{r.median_share:.3f}')
    rows.append(' & '.join(cells) + r' \\')
out['fevd'] = '\n'.join(rows)
# band row for the gap and salience only
gapb = [(h, fv[(fv.h==h)&(fv.variable=='g_meandec_cpi')].iloc[0]) for h in [0,4,20]]
out['fevd_bands'] = '; '.join(f'$h={h}$ [{r.p16:.3f}, {r.p84:.3f}]' for h,r in gapb)

# ---- C.6 diagnostics -----------------------------------------------------
adf = pd.read_excel(X,'tab_adf_v3'); vif = pd.read_excel(X,'tab_vif_v3')
nice = {'q_t_commod_yoy':'$q_t$ commodity prices','a_t_kilian':'$a_t$ global activity',
        'pi_t_rpi':'$\\pi_t$ RPI inflation','s_t_asinh100':'$s_t$ signed salience',
        'D_t_std_dec_h10':'$D_t$ disagreement','g_meandec_cpi':'$g_t$ expectations gap',
        'eri_t_yoy':'$e_t$ sterling ERI'}
v6 = vif[vif.system=='six'].set_index('variable')
rows = []
for _, r in adf.iterrows():
    vr = v6.loc[r.variable] if r.variable in v6.index else None
    rows.append(' & '.join([
        nice[r.variable], f'${r.adf:.3f}$', f'{r.pvalue:.3f}', f'{int(r.lags)}',
        ('yes' if r.reject_5pc else 'no'),
        (f'{vr.R2_on_others:.3f}' if vr is not None else '---'),
        (f'{vr.VIF:.2f}' if vr is not None else '---'),
    ]) + r' \\')
out['diag'] = '\n'.join(rows)

corr = pd.read_excel(X,'tab_corr_v3').set_index('Unnamed: 0')
rows = []
for a in vars_:
    cells = [nice[a]]
    for bcol in vars_:
        cells.append('$1$' if a==bcol else f'${corr.loc[a,bcol]:+.3f}$')
    rows.append(' & '.join(cells) + r' \\')
out['corr'] = '\n'.join(rows)

lo = pd.read_excel(X,'tab_lagorder_v3')
ld = pd.read_excel(X,'ladder_v3')
rows=[]
for _, r in ld.iterrows():
    rows.append(' & '.join([r.label.replace('_','\\_'), f'{int(r.kept)}', f'{int(r.draws):,}',
        f'{100*r.acceptance_rate:.3f}', f'{r.fevd_s_h0:.3f}', f(r.gap_h0), f(r.gap_h4)])+r' \\')
out['ladder'] = '\n'.join(rows)

for k, v in out.items():
    print('%'+'='*66); print('% BLOCK: '+k); print('%'+'='*66); print(v); print()
