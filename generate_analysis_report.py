import pathlib
import yaml
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from tqdm import tqdm
import argparse

# --- Configuration & Constants ---
plt.style.use('seaborn-v0_8-whitegrid')
DBSCAN_MIN_SAMPLES = 3
# Behavioral states from ant_simulation.py
STATE_RETURNING = 0
STATE_EXPLORING = 1
STATE_ARRESTED = 2

# --- Core Data Loading and Metric Calculation ---

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
    """Calculates key performance metrics, including all three behavioral state proportions."""
    metrics = {}
    positions = data['positions']
    states = data['behavioural_states']
    num_time_steps, num_ants = states.shape

    # Calculate proportions of each state over time
    metrics['proportion_returning'] = np.sum(states == STATE_RETURNING, axis=1) / num_ants
    metrics['proportion_exploring'] = np.sum(states == STATE_EXPLORING, axis=1) / num_ants
    metrics['proportion_arrested'] = np.sum(states == STATE_ARRESTED, axis=1) / num_ants

    # Calculate clustering metrics based on arrested ants
    largest_cluster_size = []
    for t in range(num_time_steps):
        arrested_mask = (states[t] == STATE_ARRESTED)
        arrested_positions = positions[t][arrested_mask]
        if arrested_positions.shape[0] >= DBSCAN_MIN_SAMPLES:
            db = DBSCAN(eps=dbscan_eps, min_samples=DBSCAN_MIN_SAMPLES).fit(arrested_positions)
            labels = db.labels_
            unique_labels = set(labels)
            if len(unique_labels) > (1 if -1 in labels else 0):
                cluster_sizes = [np.sum(labels == i) for i in unique_labels if i != -1]
                largest_cluster_size.append(max(cluster_sizes))
            else:
                largest_cluster_size.append(0)
        else:
            largest_cluster_size.append(0)

    metrics['largest_cluster_prop'] = np.array(largest_cluster_size) / num_ants
    return metrics

# --- Analysis and Plotting ---

def create_comprehensive_report(df: pd.DataFrame, output_dir: pathlib.Path):
    """
    Generates a 2x2 figure showing mean/std phase diagrams and temporal dynamics
    of behavioral states for representative dispersed and emergent regimes.
    """
    agg_df = df.groupby(['decay_rate', 'deposition_rate'])['final_largest_cluster_prop'].agg(['mean', 'std']).reset_index()

    fig, axes = plt.subplots(2, 2, figsize=(20, 18))
    fig.suptitle(f'Comprehensive Analysis of Collective Dynamics for {output_dir.name}', fontsize=22, y=1.03)

    # Top-Left: Heatmap of Mean Largest Cluster Fraction
    ax_mean = axes[0, 0]
    heatmap_mean_pivot = agg_df.pivot(index='decay_rate', columns='deposition_rate', values='mean')
    heatmap_mean_pivot.sort_index(ascending=False, inplace=True)
    sns.heatmap(heatmap_mean_pivot, annot=True, fmt=".2f", cmap="viridis", linewidths=.5,
                cbar_kws={'label': 'Mean Largest Cluster Fraction'}, ax=ax_mean)
    ax_mean.set_title('Phase Diagram: Mean Final Cluster Size', fontsize=16)
    ax_mean.set_xlabel('Pheromone Deposition Rate', fontsize=12)
    ax_mean.set_ylabel('Pheromone Decay Rate (Persistence)', fontsize=12)

    # Top-Right: Heatmap of Std Dev of Largest Cluster Fraction
    ax_std = axes[0, 1]
    heatmap_std_pivot = agg_df.pivot(index='decay_rate', columns='deposition_rate', values='std')
    heatmap_std_pivot.sort_index(ascending=False, inplace=True)
    sns.heatmap(heatmap_std_pivot, annot=True, fmt=".2f", cmap="magma", linewidths=.5,
                cbar_kws={'label': 'Std Dev of Largest Cluster Fraction'}, ax=ax_std)
    ax_std.set_title('Phase Diagram: Cluster Size Variability (Std. Dev.)', fontsize=16)
    ax_std.set_xlabel('Pheromone Deposition Rate', fontsize=12)
    ax_std.set_ylabel('Pheromone Decay Rate (Persistence)', fontsize=12)

    # --- Find Representative Parameters ---
    # Dispersed: low mean, low std
    dispersed_params = agg_df.loc[(agg_df['mean'] < 0.1) & (agg_df['std'] < 0.05)].sort_values('mean').iloc[0]
    # Emergent: final cluster size is significant but not saturated (e.g., 0.4-0.8), and has high variance
    emergent_candidates = agg_df[(agg_df['mean'].between(0.4, 0.8))]
    if not emergent_candidates.empty:
        emergent_params = emergent_candidates.loc[emergent_candidates['std'].idxmax()]
    else: # Fallback to highest std dev if no candidates are found
        emergent_params = agg_df.loc[agg_df['std'].idxmax()]


    # Bottom-Left: Temporal Dynamics of Dispersed Regime
    ax_dispersed = axes[1, 0]
    plot_temporal_dynamics_for_params(df, dispersed_params, ax_dispersed, 'Temporal Dynamics: Dispersed Regime')

    # Bottom-Right: Temporal Dynamics of Emergent Regime
    ax_emergent = axes[1, 1]
    plot_temporal_dynamics_for_params(df, emergent_params, ax_emergent, 'Temporal Dynamics: Emergent/Transition Regime')

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    report_filename = output_dir / "comprehensive_analysis_report_with_states.png"
    fig.savefig(report_filename, dpi=300, bbox_inches='tight')
    print(f"\nNew comprehensive analysis report saved to: {report_filename}")
    plt.show()
    plt.close(fig)

def plot_temporal_dynamics_for_params(df: pd.DataFrame, params: pd.Series, ax: plt.Axes, title: str):
    """
    Helper function to plot the mean and std dev over time for all behavioral states
    and the largest cluster size for a specific parameter set.
    """
    decay = params['decay_rate']
    depo = params['deposition_rate']
    subset_df = df[(df['decay_rate'] == decay) & (df['deposition_rate'] == depo)]
    time_array = subset_df['time'].iloc[0]

    # Data to plot: name, color, linestyle
    plot_specs = [
        ('timeseries_largest_cluster_prop', 'black', '-', 'Largest Cluster'),
        ('timeseries_arrested', 'red', '-', 'Arrested'),
        ('timeseries_exploring', 'green', '--', 'Exploring'),
        ('timeseries_returning', 'blue', ':', 'Returning')
    ]

    for key, color, style, label in plot_specs:
        all_series = np.stack(subset_df[key].values)
        mean_series = np.mean(all_series, axis=0)
        std_series = np.std(all_series, axis=0)
        
        ax.plot(time_array, mean_series, label=label, color=color, linestyle=style, linewidth=2)
        ax.fill_between(time_array, mean_series - std_series, mean_series + std_series,
                        alpha=0.15, color=color)

    ax.set_title(f"{title}\n(decay={decay}, depo={depo})", fontsize=16)
    ax.set_xlabel('Simulation Time (s)', fontsize=12)
    ax.set_ylabel('Proportion of Ants', fontsize=12)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=10)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)

def main(experiment_dir: pathlib.Path):
    """Main analysis function."""
    try:
        latest_run_dir = sorted([d for d in experiment_dir.iterdir() if d.is_dir() and d.name[0].isdigit()])[-1]
    except IndexError:
        print(f"Error: No valid multirun subdirectories found in {experiment_dir}")
        return

    print(f"Analyzing experiment run in: {latest_run_dir}")
    
    experiment_name = experiment_dir.name
    output_dir = pathlib.Path(f"analysis/{experiment_name}")
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Analysis outputs will be saved to: {output_dir}")

    run_paths = [d for d in latest_run_dir.iterdir() if d.is_dir() and d.name.isdigit()]
    all_results = []

    for run_path in tqdm(run_paths, desc="Processing runs"):
        metadata, data = load_run_data(run_path)
        if metadata is None: continue
            
        params = metadata['params']
        ant_length = params.get('ant_length', 2.5)
        dbscan_eps = ant_length * 1.5
        metrics = calculate_metrics(data, params, dbscan_eps)

        result = {
            'seed': params['seed'],
            'deposition_rate': round(params['pheromones']['pheromone_deposition_rate'], 3),
            'decay_rate': params['pheromones']['pheromone_decay_rate'],
            'final_largest_cluster_prop': metrics['largest_cluster_prop'][-1],
            'timeseries_largest_cluster_prop': metrics['largest_cluster_prop'],
            'timeseries_arrested': metrics['proportion_arrested'],
            'timeseries_exploring': metrics['proportion_exploring'],
            'timeseries_returning': metrics['proportion_returning'],
            'time': data['time']
        }
        all_results.append(result)

    if not all_results:
        print("No results found. Check the experiment directory structure.")
        return

    df = pd.DataFrame(all_results)
    create_comprehensive_report(df, output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a comprehensive analysis report for an ant clustering sweep.")
    parser.add_argument("experiment_dir", type=pathlib.Path, 
                        help="Path to the experiment's root directory (e.g., /data/michael/ants/multirun/stigmergy/E3_Transition_Search).")
    args = parser.parse_args()
    
    main(args.experiment_dir)