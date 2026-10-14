"""Run the released manuscript scripts without machine-specific paths."""
import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
FIGURES = {
    'fig2': 'generate_borehole_figure.py',
    'fig3': 'generate_b10_nmr_figure.py',
    'fig4': 'generate_b10_figure.py',
    'fig5': 'generate_predictive_figure.py',
    'fig6': 'generate_regional_figure.py',
    'fig7': 'generate_split_figure.py',
    'fig8': 'generate_fusion_figure.py',
    'fig9': 'generate_ablation_figure.py',
    'fig10': 'generate_computation_figure.py',
}
TABLES = ['generate_result_tables.py', 'generate_completed_tables.py']


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0] in ('train', 'infer', 'reference', 'demo'):
        from .model_cli import main as model_main
        return model_main(argv)
    parser = argparse.ArgumentParser(description=__doc__,
        epilog='Model commands: train, infer, reference, demo. Use COMMAND --help for details.')
    parser.add_argument('command', choices=['all', 'figures', 'tables'])
    parser.add_argument('--only', choices=list(FIGURES), help='Rebuild one experimental figure.')
    parser.add_argument('--data-dir', type=Path, default=ROOT / 'data')
    parser.add_argument('--out', type=Path, default=ROOT / 'outputs')
    parser.add_argument('--font', default='Times New Roman',
                        help='Use the manuscript font, or explicitly select an installed font.')
    args = parser.parse_args(argv)
    if args.only and args.command != 'figures':
        parser.error('--only requires the figures command')
    if not args.data_dir.is_dir():
        parser.error(f'Data directory not found: {args.data_dir}')
    from matplotlib import font_manager
    try:
        font_manager.findfont(font_manager.FontProperties(family=args.font),
                              fallback_to_default=False)
    except ValueError:
        parser.error(f'Font {args.font!r} is not installed. Install it or pass '
                     '--font "DejaVu Serif" explicitly.')
    output = args.out.resolve()
    data = args.data_dir.resolve()
    if output == data or data in output.parents:
        parser.error('--out must be outside the input data directory')
    for name in ['figures', 'tables', 'macros']:
        (output / name).mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update(RECODIF_DATA_DIR=str(data), RECODIF_OUTPUT_DIR=str(output),
               RECODIF_FONT=args.font, PYTHONUTF8='1')
    jobs = []
    if args.command in ('all', 'tables'):
        jobs.extend(TABLES)
        for name in ['Table_Data.tex', 'Table_Baselines.tex']:
            shutil.copy2(ROOT / 'assets' / name, output / 'tables' / name)
    if args.command in ('all', 'figures'):
        jobs.extend([FIGURES[args.only]] if args.only else FIGURES.values())
        if not args.only:
            shutil.copy2(ROOT / 'assets/Fig1_Framework.pdf', output / 'figures/Fig1_Framework.pdf')
    for script in jobs:
        print(f'Running {script}', flush=True)
        result = subprocess.run([sys.executable, str(ROOT / 'scripts' / script)], env=env)
        if result.returncode:
            print(f'Failed: {script} (exit {result.returncode})', file=sys.stderr)
            return result.returncode
    print(f'Completed. Outputs: {output}')
    return 0
