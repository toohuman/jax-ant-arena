import pathlib
import yaml
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from tqdm import tqdm
import argparse

# DBSCAN parameters for cluster detection
# Min Samples: The number of samples in a neighborhood for a point to be considered as a core point.
# A value of 3 is more sensitive to smaller, emerging clusters.
DBSCAN_MIN_SAMPLES = 3

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

def calculate_metrics(data, params, dbscan_eps):
    """
    Calculates key performance metrics from raw simulation data.
    The clustering analysis (DBSCAN) is performed ONLY on arrested ants
    to provide a direct measure of the pheromone-driven aggregation.
    """
    metrics = {}
    positions = data['positions'] # Shape: (time, num_ants, 2)
    states = data['behavioural_states'] # Shape: (time, num_ants)
    num_time_steps, num_ants = states.shape

    # 1. Proportion of arrested ants over time
    STATE_ARRESTED = 2 # From ant_simulation.py
    arrested_ants = np.sum(states == STATE_ARRESTED, axis=1)
    metrics['proportion_arrested'] = arrested_ants / num_ants

    # 2. Cluster analysis using DBSCAN on ARRESTED ANTS ONLY
    num_clusters = []
    largest_cluster_size = []

    for t in range(num_time_steps):
        # Filter for arrested ants at this timestep
        arrested_mask = (states[t] == STATE_ARRESTED)
        arrested_positions = positions[t][arrested_mask]

        # Only run DBSCAN if there are enough arrested ants to form a cluster
        if arrested_positions.shape[0] >= DBSCAN_MIN_SAMPLES:
            db = DBSCAN(eps=dbscan_eps, min_samples=DBSCAN_MIN_SAMPLES).fit(arrested_positions)
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
        else:
            # Not enough arrested ants to form any clusters
            num_clusters.append(0)
            largest_cluster_size.append(0)

    metrics['num_clusters'] = np.array(num_clusters)
    metrics['largest_cluster_prop'] = np.array(largest_cluster_size) / num_ants

    return metrics

def plot_timeseries(df: pd.DataFrame, output_dir: pathlib.Path):
    """Plots the time-series of arrested proportion for key runs."""
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 7))

    if df.empty:
        print("Warning: DataFrame is empty, cannot plot timeseries.")
        return

    # This is more robust than hard-coding parameter values
    min_arrest_run = df.loc[df['final_arrested_prop'].idxmin()]
    max_arrest_run = df.loc[df['final_arrested_prop'].idxmax()]

    # Plotting
    ax.plot(min_arrest_run['time'], min_arrest_run['timeseries_arrested'],
            label=f"Min Clustering (final={min_arrest_run['final_arrested_prop']:.2f})\n"
                  f"decay={min_arrest_run['decay_rate']}, depo={min_arrest_run['deposition_rate']}")


    ax.plot(max_arrest_run['time'], max_arrest_run['timeseries_arrested'],
            label=f"Max Clustering (final={max_arrest_run['final_arrested_prop']:.2f})\n"
                  f"decay={max_arrest_run['decay_rate']}, depo={max_arrest_run['deposition_rate']}",
            linestyle='--')

    ax.set_title('Temporal Dynamics of Clustering Regimes')
    ax.set_xlabel('Simulation Time (s)')
    ax.set_ylabel('Proportion of Arrested Ants')
    ax.set_ylim(0, 1.05)
    ax.legend()
    fig.tight_layout()
    timeseries_filename = output_dir / "timeseries_comparison.png"
    fig.savefig(timeseries_filename, dpi=300)
    print(f"Timeseries comparison plot saved to: {timeseries_filename}")
    plt.close(fig) # Close figure to free memory

def main(experiment_dir: pathlib.Path):
    """Main analysis function."""
    # Find the latest run automatically if multiple exist, or specify one.
    try:
        latest_run_dir = sorted([d for d in experiment_dir.iterdir() if d.is_dir() and d.name[0].isdigit()])[-1]
    except IndexError:
        print(f"Error: No subdirectories found in {experiment_dir}")
        return

    print(f"Analyzing experiment run in: {latest_run_dir}")
    
    # --- Setup Output Directory ---
    experiment_name = experiment_dir.name
    output_dir = pathlib.Path(f"analysis/{experiment_name}")
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Analysis outputs will be saved to: {output_dir}")

    run_paths = [d for d in latest_run_dir.iterdir() if d.is_dir() and d.name.isdigit()]
    
    all_results = []

    for run_path in tqdm(run_paths, desc="Processing runs"):
        metadata, data = load_run_data(run_path)
        if metadata is None:
            continue
            
        params = metadata['params']
        # Use a physically meaningful epsilon for DBSCAN based on ant size
        ant_length = params.get('ant_length', 2.5) # Default to 2.5 if not in metadata
        dbscan_eps = ant_length * 1.5 # Use 1.5x ant_length for a more forgiving cluster radius
        metrics = calculate_metrics(data, params, dbscan_eps)

        result = {
            'seed': params['seed'],
            'deposition_rate': round(params['pheromones']['pheromone_deposition_rate'], 3),
            'threshold': round(params['pheromones']['pheromone_threshold'], 3),
            'decay_rate': params['pheromones']['pheromone_decay_rate'],
            'final_arrested_prop': metrics['proportion_arrested'][-1],
            'final_largest_cluster_prop': metrics['largest_cluster_prop'][-1],
            'timeseries_arrested': metrics['proportion_arrested'],
            'time': data['time']
        }
        all_results.append(result)

    if not all_results:
        print("No results found. Check the experiment directory structure.")
        return

    # --- Create and Aggregate DataFrame ---
    df = pd.DataFrame(all_results)

    # Plot and save the time-series comparison into the analysis directory
    plot_timeseries(df, output_dir)
    
    # Aggregate results by taking the mean across seeds
    agg_df = df.groupby(['decay_rate', 'deposition_rate']).mean().reset_index()

    # --- Create the Heatmap for Arrested Proportion ---
    heatmap_arrested = agg_df.pivot(
        index='decay_rate',
        columns='deposition_rate',
        values='final_arrested_prop'
    )
    heatmap_arrested.sort_index(ascending=False, inplace=True)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    
    sns.heatmap(
        heatmap_arrested,
        annot=True,
        fmt=".2f",
        cmap="viridis",
        linewidths=.5,
        cbar_kws={'label': 'Proportion of Arrested Ants'},
        ax=ax1
    )
    ax1.set_title('Phase Diagram 1: Proportion of Arrested Ants')
    ax1.set_ylabel('Pheromone Decay Rate (Persistence)')

    # --- Create the Heatmap for Largest Cluster Proportion ---
    heatmap_cluster = agg_df.pivot(
        index='decay_rate',
        columns='deposition_rate',
        values='final_largest_cluster_prop'
    )
    heatmap_cluster.sort_index(ascending=False, inplace=True)

    sns.heatmap(
        heatmap_cluster,
        annot=True,
        fmt=".2f",
        cmap="magma", # Use a different colormap to distinguish
        linewidths=.5,
        cbar_kws={'label': 'Proportion of Ants in Largest Cluster'},
        ax=ax2
    )
    ax2.set_title('Phase Diagram 2: Spatial Clustering (Largest Cluster)')
    ax2.set_xlabel('Pheromone Deposition Rate')
    ax2.set_ylabel('Pheromone Decay Rate (Persistence)')

    plt.tight_layout()

    # Save the plot
    output_filename = output_dir / "phase_diagrams.png"
    plt.savefig(output_filename, dpi=300)
    print(f"\nPhase diagrams saved to: {output_filename}")
    plt.show()
    plt.close(fig)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyse a Hydra multirun sweep for ant clustering.")
    parser.add_argument("experiment_dir", type=pathlib.Path, 
                        help="Path to the experiment's root directory (e.g., /data/michael/ants/multirun/stigmergy/E3_Transition_Search).")
    args = parser.parse_args()
    
    main(args.experiment_dir)