import pathlib
import yaml
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from tqdm import tqdm
import os

# --- Configuration ---
# ⚠️ UPDATE THIS PATH to your specific multirun output directory
EXPERIMENT_ROOT = pathlib.Path("/data/michael/ants/multirun/stigmergy/E1_Deposition-Decay_Sweep/")
# Find the latest run automatically if multiple exist, or specify one.
try:
    LATEST_RUN_DIR = sorted([d for d in EXPERIMENT_ROOT.iterdir() if d.is_dir()])[-1]
except IndexError:
    print(f"Error: No subdirectories found in {EXPERIMENT_ROOT}")
    exit()

# DBSCAN parameters for cluster detection
# Epsilon: max distance between two samples for one to be considered as in the neighborhood of the other.
# This should be roughly the interaction distance, e.g., 2-3x ant length.
# Let's assume ant_length is ~0.05 based on typical configs.
DBSCAN_EPS = 0.15
# Min Samples: The number of samples in a neighborhood for a point to be considered as a core point.
# This is the minimum size of a "cluster".
DBSCAN_MIN_SAMPLES = 4

def load_run_data(run_path: pathlib.Path):
    """Loads metadata and simulation data from a single run directory."""
    metadata_file = run_path / "metadata.yaml"
    data_file = run_path / "simulation_data.npz"

    if not metadata_file.exists() or not data_file.exists():
        return None, None

    with open(metadata_file, 'r') as f:
        metadata = yaml.safe_load(f)

    data = np.load(data_file)
    return metadata, data

def calculate_metrics(data, params):
    """Calculates key performance metrics from raw simulation data."""
    metrics = {}
    positions = data['positions'] # Shape: (time, num_ants, 2)
    states = data['behavioural_states'] # Shape: (time, num_ants)
    num_time_steps, num_ants = states.shape

    # 1. Proportion of arrested ants over time
    STATE_ARRESTED = 2 # From ant_simulation.py
    arrested_ants = np.sum(states == STATE_ARRESTED, axis=1)
    metrics['proportion_arrested'] = arrested_ants / num_ants

    # 2. Cluster analysis using DBSCAN
    num_clusters = []
    largest_cluster_size = []

    for t in range(num_time_steps):
        # Run DBSCAN on the positions at this timestep
        db = DBSCAN(eps=DBSCAN_EPS, min_samples=DBSCAN_MIN_SAMPLES).fit(positions[t])
        labels = db.labels_

        # "-1" label is for noise (unclustered ants)
        unique_labels = set(labels)
        n_clusters_t = len(unique_labels) - (1 if -1 in labels else 0)
        num_clusters.append(n_clusters_t)

        if n_clusters_t > 0:
            # Count the size of each cluster
            cluster_sizes = [np.sum(labels == i) for i in unique_labels if i != -1]
            largest_cluster_size.append(max(cluster_sizes))
        else:
            largest_cluster_size.append(0)

    metrics['num_clusters'] = np.array(num_clusters)
    metrics['largest_cluster_prop'] = np.array(largest_cluster_size) / num_ants

    return metrics

def plot_timeseries(df: pd.DataFrame, output_dir: pathlib.Path):
    """Plots the time-series of arrested proportion for key runs."""
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 7))

    # Find the parameters for key behavioural regimes from the dataframe
    # 1. Dispersed/Emergent (bottom-left corner of your heatmap)
    dispersed_run = df[(df['decay_rate'] == 0.995) & (df['deposition_rate'] == 0.5)].iloc[0]

    # 2. Frozen/Saturated (top-right corner)
    frozen_run = df[(df['decay_rate'] == 0.9995) & (df['deposition_rate'] == 4.0)].iloc[0]

    # Plotting
    ax.plot(dispersed_run['time'], dispersed_run['timeseries_arrested'],
            label=f"Emergent: decay={dispersed_run['decay_rate']}, depo={dispersed_run['deposition_rate']}")

    ax.plot(frozen_run['time'], frozen_run['timeseries_arrested'],
            label=f"Frozen: decay={frozen_run['decay_rate']}, depo={frozen_run['deposition_rate']}",
            linestyle='--')

    ax.set_title('Temporal Dynamics of Clustering Regimes')
    ax.set_xlabel('Simulation Time (s)')
    ax.set_ylabel('Proportion of Arrested Ants')
    ax.set_ylim(0, 1.05)
    ax.legend()
    fig.tight_layout()
    timeseries_filename = output_dir / "timeseries_comparison.png"
    fig.savefig(timeseries_filename, dpi=300)
    print(f"✅ Timeseries comparison plot saved to: {timeseries_filename}")

def main():
    """Main analysis function."""
    print(f"🔍 Analyzing experiment run in: {LATEST_RUN_DIR}")
    run_paths = [d for d in LATEST_RUN_DIR.iterdir() if d.is_dir() and d.name.isdigit()]
    
    all_results = []

    for run_path in tqdm(run_paths, desc="Processing runs"):
        metadata, data = load_run_data(run_path)
        if metadata is None:
            continue
            
        params = metadata['params']
        metrics = calculate_metrics(data, params)

        # We are interested in the final state of the system for the heatmap
        # And the full timeseries for later plots
        result = {
            'seed': params['seed'],
            'deposition_rate': params['pheromones']['pheromone_deposition_rate'],
            'decay_rate': params['pheromones']['pheromone_decay_rate'],
            'final_arrested_prop': metrics['proportion_arrested'][-1],
            'final_largest_cluster_prop': metrics['largest_cluster_prop'][-1],
            'timeseries_arrested': metrics['proportion_arrested'],
            'time': data['time']
        }
        all_results.append(result)

    if not all_results:
        print("No results found. Check the EXPERIMENT_ROOT path and directory structure.")
        return

    # --- Create and Aggregate DataFrame ---
    df = pd.DataFrame(all_results)

    # Plot and save the time-series comparison before aggregation
    plot_timeseries(df, LATEST_RUN_DIR)
    
    # Aggregate results by taking the mean across seeds
    agg_df = df.groupby(['decay_rate', 'deposition_rate']).mean().reset_index()

    # --- Create the Heatmap (Phase Diagram) ---
    heatmap_data = agg_df.pivot(
        index='decay_rate', 
        columns='deposition_rate', 
        values='final_arrested_prop'
    )
    # Sort index for a clean plot
    heatmap_data.sort_index(ascending=False, inplace=True)

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        heatmap_data,
        annot=True,
        fmt=".2f",
        cmap="viridis",
        linewidths=.5,
        cbar_kws={'label': 'Proportion of Arrested Ants at t=10000s'}
    )
    plt.title('Clustering Behaviour vs. Pheromone Dynamics')
    plt.xlabel('Pheromone Deposition Rate')
    plt.ylabel('Pheromone Decay Rate (Persistence)')
    plt.tight_layout()
    
    # Save the plot
    output_directory = "analysis/E1_Deposition-Decay_Sweep"
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    output_filename = os.path.join(output_directory, "phase_diagram_arrested_prop.png")
    plt.savefig(output_filename, dpi=300)
    print(f"\nPhase diagram saved to: {output_filename}")
    plt.show()

if __name__ == "__main__":
    main()