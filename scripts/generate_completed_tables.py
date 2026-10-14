"""Tabulate the updated experiment results without altering source CSVs."""
from completed_results import *

body = []
for method in METHODS + ['DLEMLR']:
    data = primary(method)
    assert len(data) == 5, (method, len(data))
    body.append([method_label(method),
        tex_mean_sd(data, 'normalized_echo_rmse'),
        tex_mean_sd(data, 'empirical_crps'),
        tex_mean_sd(data, 'coverage_fraction', 2, 100),
        tex_mean_sd(data, 'normalized_interval_width'),
        tex_mean_sd(data, 'multivariate_energy_score')])
table('Table_B10.tex',
      r'B10 predictions, averaged over depths then seeds (mean $\pm$ sample standard deviation, five seeds). Coverage and width refer to central 90\% intervals; amplitude metrics use $s_c=10$. DLEMLR gives point predictions.',
      'tab:b10-results',
      ['Method', 'RMSE', 'CRPS', r'Coverage (\%)', 'Width', 'ES'], body)

ablations = variants()
body = [[VARIANTS[k], tex_mean_sd(data, 'normalized_echo_rmse'),
         tex_mean_sd(data, 'crps'), tex_mean_sd(data, 'coverage90', 2, 100),
         tex_mean_sd(data, 'width90'), tex_mean_sd(data, 'regional_energy_v')]
        for k, data in ablations.items()]
table('Table_Ablation.tex',
      r'Ablations at 7.0 m (mean $\pm$ sample SD, five seeds). Coverage and width use central 90\% intervals. Regional discrepancy uses stored amplitude units; other amplitude metrics use $s_c=10$.',
      'tab:ablation-results',
      ['Variant', 'RMSE', 'CRPS', r'Coverage (\%)', 'Width', r'$\widehat{\mathcal D}_{E,V}$'], body)

costs = rows('runtime_results.csv')
body = []
tasks = {'ReDif':'Joint posterior', 'DDRM':'Joint posterior', 'DPS':'Joint posterior',
         'CSGM':'Joint posterior', 'DAPS':'Joint posterior', 'DLEMLR':'Joint point',
         'DRECT':'1D point', 'DREAM':'Axis marginal', 'bNMR_McMC':'Native 1D posterior'}
for method in METHODS + ['DLEMLR', 'DRECT', 'DREAM', 'bNMR_McMC']:
    data = [r for r in costs if r['method'] == method]
    assert len(data) == 30
    q1, med, q3 = np.quantile(values(data, 'total_inference_s'), [.25, .5, .75])
    memory = np.median(values(data, 'peak_memory_mb'))
    label = method_label(method)
    body.append([label, tasks[method], data[0]['draw_count'],
                 f'{med:.3f} [{q1:.3f}, {q3:.3f}]', f'{memory:.1f}',
                 data[0]['likelihood_calls'], data[0]['gradient_calls']])
table('Table_Runtime.tex',
      r'CPU cost per evidence set. Time is median [first, third quartile] over 30 repeats after ten warm-ups; memory is median fresh-process peak. $N_{\rm like}$/$N_{\rm grad}$ count likelihood/gradient calls.',
      'tab:runtime-results',
      ['Method', 'Output', 'Draws', 'Time (s)', 'Memory (MiB)', r'$N_{\rm like}$', r'$N_{\rm grad}$'], body)

descriptions = {
    'NUM_early_refinement': ('Early refinement', '$t=0.70,0.35$'),
    'NUM_late_refinement': ('Late refinement', '$t=0.25,0.08$'),
    'NUM_alternative_split_order': ('Split order', '$T_2$ then $T_1$'),
    'NUM_reverse_step_sensitivity': ('Reverse step', r'$\Delta t=0.004$'),
    'NUM_auxiliary_duration_sensitivity': ('Auxiliary duration', r'$\tau=0.5$'),
    'NUM_quadrature_tolerance_sensitivity': ('Quadrature', '$10^{-8}$'),
    'NUM_terminal_initialization': ('Initialization', 'Matched terminal profile'),
    'NUM_positive_stopping_time': ('Positive stop', r'$\epsilon=0.01$, no lift'),
    'NUM_final_transition_to_zero': ('Final transition', r'$\epsilon=0.001$, with lift'),
}
body = []
for r in rows('numerical_and_schedule_results.csv'):
    label, setting = descriptions[r['case_id']]
    body.append([label, setting, f"{float(r['crps']):.5f}",
                 f"{float(r['reference_energy_v']):.5f}", f"{float(r['mass_residual']):.1g}"])
table('Table_Numerical.tex',
      r'Numerical sensitivity at 7.0 m (seed 1001; 512 draws). Defaults: split times $0.50/0.20$, reverse step 0.002, auxiliary duration 1.0, quadrature tolerance $10^{-10}$. Residual measures parent--child mass error.',
      'tab:numerical-results',
      ['Variant', 'Setting', 'CRPS', r'$\widehat{\mathcal D}_{E,V}$', 'Residual'], body)

def metric(method, key):
    return mean_sd(primary(method), key)[0]
full = mean_sd(ablations['full_shared_stochastic'], 'regional_energy_v')[0]
noagg = mean_sd(ablations['no_aggregation_shared_stochastic'], 'regional_energy_v')[0]
det = mean_sd(ablations['aggregation_deterministic_split'], 'crps')[0]
full_crps = mean_sd(ablations['full_shared_stochastic'], 'crps')[0]
commands = {
    'BtenCRPS': f"{metric('ReDif', 'empirical_crps'):.4f}",
    'BtenCoverage': f"{100*metric('ReDif', 'coverage_fraction'):.2f}",
    'BtenRMSE': f"{metric('ReDif', 'normalized_echo_rmse'):.4f}",
    'BtenCRPSGainDDRM': f"{100*(1-metric('ReDif','empirical_crps')/metric('DDRM','empirical_crps')):.2f}",
    'BtenCRPSGainCSGM': f"{100*(1-metric('ReDif','empirical_crps')/metric('CSGM','empirical_crps')):.2f}",
    'AblationRegionalReduction': f'{100*(1-full/noagg):.2f}',
    'AblationSplitCRPSReduction': f'{100*(1-full_crps/det):.2f}',
    'ReDifCostMedian': f"{np.median(values([r for r in costs if r['method']=='ReDif'],'total_inference_s')):.2f}",
}
(MACRODIR / 'completed-result-values.tex').write_text(
    '% Derived from ReDif_实验CSV.\n' + '\n'.join(rf'\newcommand{{\{k}}}{{{v}}}' for k,v in commands.items()) + '\n',
    encoding='utf-8')
print('Generated four result tables and current-result values.')
