import re
from pathlib import Path
import matplotlib.pyplot as plt

# Parse cold-start AUC from rigorous summary
rig = Path('rigorous_metrics_all.txt').read_text(encoding='utf-8', errors='ignore')
mt = re.search(r'^AUC:\s*([0-9.]+)\s*\+\-', rig, flags=re.M)
if not mt:
    raise SystemExit('Could not find cold-start AUC in rigorous_metrics_all.txt')
cold_auc = float(mt.group(1))

# Parse transductive test AUC from get_metrics output
txt = Path('real_metrics_output.txt').read_text(encoding='utf-16', errors='ignore')
# fallback if file is not utf-16
if 'TEST SET METRICS' not in txt:
    txt = Path('real_metrics_output.txt').read_text(encoding='utf-8', errors='ignore')

m = re.search(r'TEST SET METRICS \(FINAL PERFORMANCE\).*?AUC-ROC:\s*([0-9.]+)', txt, flags=re.S)
if not m:
    # allow generic AUC-ROC capture near TEST block
    m = re.search(r'TEST SET METRICS.*?AUC-ROC:\s*([0-9.]+)', txt, flags=re.S)
if not m:
    raise SystemExit('Could not find transductive test AUC in real_metrics_output.txt')
trans_auc = float(m.group(1))

out = Path('final_plot')
out.mkdir(exist_ok=True)

# Save metrics summary
summary = out / 'metrics_terminal_summary.txt'
summary.write_text(
    'REAL METRICS SUMMARY\n'
    '====================\n'
    f'Cold-Start AUC (10-seed mean): {cold_auc:.4f}\n'
    f'Transductive Test AUC: {trans_auc:.4f}\n',
    encoding='utf-8'
)

# Bar plot: cold-start vs transductive
plt.figure(figsize=(7,5))
labels = ['Cold-Start (Inductive)', 'Transductive Test']
vals = [cold_auc, trans_auc]
colors = ['#1f77b4', '#ff7f0e']
plt.bar(labels, vals, color=colors)
plt.ylim(0.5, 1.0)
for i, v in enumerate(vals):
    plt.text(i, v + 0.01, f'{v:.4f}', ha='center', va='bottom', fontsize=10)
plt.ylabel('AUC-ROC')
plt.title('Cold-Start vs Transductive AUC')
plt.tight_layout()
plt.savefig(out / 'coldstart_vs_transductive_auc.png', dpi=300)
print(f'Cold-start AUC: {cold_auc:.4f}')
print(f'Transductive AUC: {trans_auc:.4f}')
print('Saved: final_plot/coldstart_vs_transductive_auc.png')
print('Saved: final_plot/metrics_terminal_summary.txt')
