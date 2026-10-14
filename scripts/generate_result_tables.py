"""Generate manuscript tables and numerical text macros from the supplied CSVs."""
from experiment_data import *

metrics=read('metrics.csv')
macro={r['method']:r for r in metrics if r['level']=='macro_site'}
supp=supplemented('performance_by_unit.csv')
energy={r['method']:float(r['multivariate_energy_score']) for r in supp if r['level']=='macro_site' and r['interval_level']=='0.9'}
def write(name,text):
    (TABLEDIR/name).write_text(text.strip()+'\n',encoding='utf-8')

lines=[r'\begin{table*}[t]',r'\centering',
       r'\caption{Predictive scores. Overall rows use equal group weights; regime rows summarize C. Coverage and width refer to central 90\% intervals; ES is the 96-dimensional energy score.}',
       r'\label{tab:predictive-results}',r'\small',r'\begin{tabular}{llrrrrrr}',r'\toprule',
       r'Ensemble & Subset & Records & RMSE & CRPS & ES & Coverage (\%) & Width\\',r'\midrule']
for method in METHODS:
    r=macro[method]
    lines.append(f"{LABELS[method]} & All records & 12 & {float(r['normalized_echo_rmse']):.5f} & {float(r['empirical_crps']):.5f} & {energy[method]:.5f} & {100*float(r['coverage_90']):.2f} & {float(r['interval_width_90']):.5f}\\\\")
lines.append(r'\midrule')
regime_stats={}
for regime,label in [('nominal','Nominal'),('low_snr','Low SNR'),('mismatch','Kernel mismatch')]:
    rr=[r for r in metrics if r['level']=='record' and r['method']=='candidate_mock' and r['regime']==regime]
    vals={k:float(np.mean(values(rr,k))) for k in ['normalized_echo_rmse','empirical_crps','coverage_90','interval_width_90']}
    regime_stats[regime]=vals
    ids={r['unit_id'] for r in rr}
    es=np.mean([float(r['multivariate_energy_score']) for r in supp if r['level']=='record' and r['method']=='candidate_mock' and r['interval_level']=='0.9' and r['unit_id'] in ids])
    lines.append(f"C & {label} & {len(rr)} & {vals['normalized_echo_rmse']:.5f} & {vals['empirical_crps']:.5f} & {es:.5f} & {100*vals['coverage_90']:.2f} & {vals['interval_width_90']:.5f}\\\\")
lines += [r'\bottomrule',r'\end{tabular}',r'\end{table*}']
write('Table_Predictive.tex','\n'.join(lines))

regions=read('toy_region_energy.csv')
dist={(r['ensemble_a'],r['ensemble_b']):float(r['energy_v']) for r in regions}
stats=split_statistics();worst_cdf=max(stats,key=lambda r:r['cdf']);worst_tv=max(stats,key=lambda r:r['tv'])
lines=[r'\begin{table*}[t]',r'\centering',
       r'\caption{Regional discrepancy (upper) and conditional allocation error (lower). CDF error is the maximum absolute bin-boundary error; TV compares bin probabilities. The lower block includes both worst-error cases.}',
       r'\label{tab:mechanism-results}',r'\small',r'\begin{tabular}{lrrr}',r'\toprule',
       r'Comparison & Draws/law & Discrepancy ($10^{-3}$ a.u.) & Ratio to $R_1$--$R_2$\\',r'\midrule']
ref=dist[('reference_ensemble_a','reference_ensemble_b')]
for a,b,label in [('coarse_path_mock','fine_aggregated_mock',r'$F_c$--$F_f$'),('coarse_path_mock','reference_ensemble_a',r'$F_c$--$R_1$'),('fine_aggregated_mock','reference_ensemble_a',r'$F_f$--$R_1$'),('reference_ensemble_a','reference_ensemble_b',r'$R_1$--$R_2$')]:
    d=dist[(a,b)];lines.append(f'{label} & 512 & {1000*d:.4f} & {d/ref:.2f}\\\\')
lines += [r'\midrule',r'Mass $M$ (a.u.) & Counts & CDF error & TV\\',r'\midrule']
for r in [stats[0],worst_cdf,worst_tv,stats[-1]]:
    lines.append(f"{r['mass']:.4f} & 1200 & {r['cdf']:.5f} & {r['tv']:.5f}\\\\")
lines += [r'\bottomrule',r'\end{tabular}',r'\end{table*}']
write('Table_Mechanisms.tex','\n'.join(lines))

base,response=fusion_summaries()
zero=sorted([r for r in response if r['recovery_time_relative_error']==0],key=lambda r:r['added_context_count'])
lines=[r'\begin{table*}[t]',r'\centering',
       r'\caption{Fusion at correct timing. Width and coverage use central 90\% intervals; gain is the CRPS reduction relative to the base case. Added observations form nested subsets.}',
       r'\label{tab:fusion-results}',r'\small',r'\begin{tabular}{rrrrrr}',r'\toprule',
       r'Added & Context & CRPS (a.u.) & Width (a.u.) & Covered & Gain (\%)\\',r'\midrule']
for r in [base]+zero:
    gain=100*(1-r['crps_mean']/base['crps_mean'])
    lines.append(f"{int(r['added_context_count'])} & {int(r['context_count'])} & {r['crps_mean']:.5f} & {r['width90_au']:.5f} & {int(round(r['coverage90']*6))}/6 & {gain:.2f}\\\\")
lines += [r'\bottomrule',r'\end{tabular}',r'\end{table*}']
write('Table_Fusion.tex','\n'.join(lines))

train=next(r for r in supplemented('training_runs.csv') if r['method']=='ReDif')
runtime=next(r for r in supplemented('runtime_results.csv') if r['method']=='ReDif')
cohort=next(r for r in supplemented('cohort_results.csv') if r['record_id']=='B10_6-4')
lines=[r'\begin{table*}[t]',r'\centering',
       r'\caption{Training and runtime settings.}',
       r'\label{tab:experimental-settings}',r'\small',r'\begin{tabularx}{\textwidth}{@{}p{0.20\textwidth}X@{}}',r'\toprule',r'Component & Setting\\',r'\midrule',
       f"Optimizer & AdamW; learning rate ${float(train['learning_rate']):.4f}$; weight decay ${float(train['weight_decay']):.4f}$; batch size {train['effective_batch_size']}; gradient clipping {train['gradient_clip_norm']}.\\\\",
       r'AdamW parameters & $(\beta_1,\beta_2)=(0.9,0.999)$; $\epsilon=10^{-8}$.\\',
       r'Schedule & 100,000 updates; 5000-update linear warm-up, then cosine decay to $2\times10^{-6}$; validation every 2000 updates.\\',
       r'Aggregation & First 10,000 updates: denoising only; then $\lambda_{\mathrm{agg}}=0.1$, teacher EMA $0.999$.\\',
       r'Tuning & Learning rates $\{1,2,3\}\times10^{-4}$ and weight decays $\{0,10^{-4}\}$; 25,000 updates/trial, seed 9001.\\',
       r'Repetitions & Training seeds 1001--1005; 128 validation and 512 test draws per evidence set.\\',
       r'Site folds & Training/validation/recalibration/test: $4/1/2/1$ sites in eight field folds; $5/1/2/1$ in nine laboratory folds.\\',
       f"Hardware & Intel Core i7-14700F ({runtime['cpu_physical_cores']} cores, {runtime['cpu_logical_processors']} threads); {runtime['memory_installed_gib']} GiB DDR5, {runtime['memory_configured_rate_mt_s']} MT/s; Windows 11.\\\\",
       r'Timing & Batch size 1; 512 draws, microbatches of 32 where applicable; 8 intra-operation/1 inter-operation thread; 10 warm-ups, 30 repeats.\\',
       r'Precision & Float32 networks; float64 likelihood and sensitive numerical operations.\\',
       r'\bottomrule',r'\end{tabularx}',r'\end{table*}']
write('Table_Settings.tex','\n'.join(lines))

c,a,b=[macro[m] for m in METHODS]
full=zero[-1]
numbers={
    'MainCRPSC':f"{float(c['empirical_crps']):.5f}",
    'MainCRPSA':f"{float(a['empirical_crps']):.5f}",
    'MainCRPSB':f"{float(b['empirical_crps']):.5f}",
    'MainCoverageC':f"{100*float(c['coverage_90']):.2f}",
    'MainCoverageA':f"{100*float(a['coverage_90']):.2f}",
    'MainCoverageB':f"{100*float(b['coverage_90']):.2f}",
    'CRPSWorseA':f"{100*(float(c['empirical_crps'])/float(a['empirical_crps'])-1):.2f}",
    'CRPSBetterB':f"{100*(1-float(c['empirical_crps'])/float(b['empirical_crps'])):.2f}",
    'WidthReductionA':f"{100*(1-float(c['interval_width_90'])/float(a['interval_width_90'])):.2f}",
    'WidthReductionB':f"{100*(1-float(c['interval_width_90'])/float(b['interval_width_90'])):.2f}",
    'SplitCDFMedian':f"{np.median([r['cdf'] for r in stats]):.5f}",
    'SplitCDFMax':f"{worst_cdf['cdf']:.5f}",
    'SplitTVMedian':f"{np.median([r['tv'] for r in stats]):.5f}",
    'SplitTVMax':f"{worst_tv['tv']:.5f}",
    'FusionBaseCRPS':f"{base['crps_mean']:.5f}",
    'FusionFullCRPS':f"{full['crps_mean']:.5f}",
    'FusionBaseWidth':f"{base['width90_au']:.5f}",
    'FusionFullWidth':f"{full['width90_au']:.5f}",
    'FusionGain':f"{100*(1-full['crps_mean']/base['crps_mean']):.2f}",
    'FusionWidthReduction':f"{100*(1-full['width90_au']/base['width90_au']):.2f}",
    'RegionalReduction':f"{100*(1-dist[('fine_aggregated_mock','reference_ensemble_a')]/dist[('coarse_path_mock','reference_ensemble_a')]):.2f}",
    'FusionMinimumGain':f"{min(100*(1-r['crps_mean']/base['crps_mean']) for r in response):.2f}",
    'FusionMaximumGain':f"{max(100*(1-r['crps_mean']/base['crps_mean']) for r in response):.2f}",
}
(MACRODIR/'result-values.tex').write_text('% Numerical values generated from the experiment CSVs.\n'+'\n'.join('\\newcommand{\\'+k+'}{'+v+'}' for k,v in numbers.items())+'\n',encoding='utf-8')
print('Generated four result/settings tables and numerical text values.')
