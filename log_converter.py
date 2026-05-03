import os

def convert(fin, fout):
    if os.path.exists(fin):
        try:
            with open(fin, 'r', encoding='utf-16') as f:
                content = f.read()
            with open(fout, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"Successfully converted {fin} to {fout}")
        except Exception as e:
            print(f"Failed to convert {fin}: {e}")
    else:
        print(f"Source file {fin} not found")

convert('ablation_results_10seeds.txt', 'ablation_results_readable.txt')
convert('diversity_results.txt', 'diversity_results_readable.txt')
