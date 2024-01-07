#!/usr/bin/env python
# coding: utf-8

# # Simple Plotting
# 

# In[ ]:


RESULTS_PATH = "../../your_sweep_results_path"

PLOT_ALL_SEEDS = False
# Full sweep
MODELS_TO_PLOT = ["gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl", "Qwen/Qwen-1_8B", "Qwen/Qwen-7B", "Qwen/Qwen-14B"]
# Minimal sweep
# MODELS_TO_PLOT = ["gpt2", "gpt2-medium", "gpt2-large"]


# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')

from IPython.display import display

import os
import glob
import json


# In[ ]:


records = []
all_results_folders = ['/'.join(e.split('/')[:-1]) for e in glob.glob(os.path.join(RESULTS_PATH, "**/*.results_summary.json"), recursive=True)]
for result_folder in set(all_results_folders):
    config_file = os.path.join(result_folder, "config.json")
    config = json.load(open(config_file, "r"))
    if config["strong_model_size"] not in MODELS_TO_PLOT:
        continue
    if 'seed' not in config:
        config['seed'] = 0
    result_filename = (config["weak_model_size"].replace('.', '_') + "_" + config["strong_model_size"].replace('.', '_') + ".results_summary.json").replace('/', '_')
    record = config.copy()
    record.update(json.load(open(config_file.replace('config.json', result_filename))))
    records.append(record)

df = pd.DataFrame.from_records(records).sort_values(['ds_name', 'weak_model_size', 'strong_model_size'])


# In[ ]:


datasets = df.ds_name.unique()
for dataset in datasets:
    cur_df = df[(df.ds_name == dataset)]
    base_df = pd.concat([
        pd.DataFrame.from_dict({"strong_model_size": cur_df['weak_model_size'].to_list(), "accuracy": cur_df['weak_acc'].to_list(), "seed": cur_df['seed'].to_list()}),
        pd.DataFrame.from_dict({"strong_model_size": cur_df['strong_model_size'].to_list(), "accuracy": cur_df['strong_acc'].to_list(), "seed": cur_df['seed'].to_list()})
    ])
    base_accuracies = base_df.groupby('strong_model_size').agg({'accuracy': 'mean', 'seed': 'count'}).sort_values('accuracy')
    base_accuracy_lookup = base_accuracies['accuracy'].to_dict()
    base_accuracies = base_accuracies.reset_index()
    base_df.reset_index(inplace=True)
    base_df['weak_model_size'] = 'ground truth'
    base_df['loss'] = 'xent'
    base_df['strong_model_accuracy'] = base_df['strong_model_size'].apply(lambda x: base_accuracy_lookup[x])

    weak_to_strong = cur_df[['weak_model_size', 'strong_model_size', 'seed'] + [e for e in cur_df.columns if e.startswith('transfer_acc')]]
    weak_to_strong = weak_to_strong.melt(id_vars=['weak_model_size', 'strong_model_size', 'seed'], var_name='loss', value_name='accuracy')
    weak_to_strong = weak_to_strong.dropna(subset=['accuracy'])
    weak_to_strong.reset_index(inplace=True)
    weak_to_strong['loss'] = weak_to_strong['loss'].str.replace('transfer_acc_', '')
    weak_to_strong['strong_model_accuracy'] = weak_to_strong['strong_model_size'].apply(lambda x: base_accuracy_lookup[x])

    # Exclude cases where the weak model is better than the strong model from PGR calculation.
    pgr_df = cur_df[(cur_df['weak_model_size'] != cur_df['strong_model_size']) & (cur_df['strong_acc'] > cur_df['weak_acc'])]
    pgr_df = pgr_df.melt(id_vars=[e for e in cur_df.columns if not e.startswith('transfer_acc')], var_name='loss', value_name='transfer_acc')
    pgr_df = pgr_df.dropna(subset=['transfer_acc'])
    pgr_df['loss'] = pgr_df['loss'].str.replace('transfer_acc_', '')
    pgr_df['pgr'] = (pgr_df['transfer_acc'] - pgr_df['weak_acc']) / (pgr_df['strong_acc'] - pgr_df['weak_acc'])

    for seed in [None] + (sorted(cur_df['seed'].unique().tolist()) if PLOT_ALL_SEEDS else []):
        plot_df = pd.concat([base_df, weak_to_strong])
        seed_pgr_df = pgr_df
        if seed is not None:
            plot_df = plot_df[plot_df['seed'] == seed]
            # We mean across seeds, this is because sometimes the weak and strong models will have run on different hardware and therefore
            # have slight differences. We want to average these out when filtering by seed.

            seed_pgr_df = pgr_df[pgr_df['seed'] == seed]

        if seed is not None or cur_df['seed'].nunique() == 1:
            plot_df = plot_df[['strong_model_accuracy', 'weak_model_size', 'loss', 'accuracy']].groupby(['strong_model_accuracy', 'weak_model_size', 'loss']).mean().reset_index().sort_values(['loss', 'weak_model_size'], ascending=False)

        print(f"Dataset: {dataset} (seed: {seed})")

        pgr_results = seed_pgr_df.groupby(['loss']).aggregate({"pgr": "median"})
        display(pgr_results)

        palette = sns.color_palette('colorblind', n_colors=len(plot_df['weak_model_size'].unique()) - 1)
        color_dict = {model: ("black" if model == 'ground truth' else palette.pop()) for model in plot_df['weak_model_size'].unique()}

        sns.lineplot(data=plot_df, x='strong_model_accuracy', y='accuracy', hue='weak_model_size', style='loss', markers=True, palette=color_dict)
        pd.plotting.table(plt.gca(), pgr_results.round(4), loc='lower right', colWidths=[0.1, 0.1], cellLoc='center', rowLoc='center')
        plt.xticks(ticks=base_accuracies['accuracy'], labels=[f"{e} ({base_accuracy_lookup[e]:.4f})" for e in base_accuracies['strong_model_size']], rotation=90)
        plt.title(f"Dataset: {dataset} (seed: {seed})")
        plt.legend(loc='upper left')
        plt.savefig(f"{dataset.replace('/', '-')}_{seed}.png", dpi=300, bbox_inches='tight')
        plt.show()

