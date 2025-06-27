import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import seaborn as sns
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import DBSCAN
from scipy.stats import sem
import matplotlib.patches as patches

# Set publication-quality defaults
plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10

class PhaseAnalyser:
    def __init__(self, data_dir, ant_length=1.0):
        self.data_dir = Path(data_dir)
        self.ant_length = ant_length
        self.cluster_threshold = 2.5 * ant_length  # Clustering distance threshold
        self.arena_radius = None  # Will be set from metadata


    @staticmethod
    def find_latest_run(base_dir):
        """Find the latest timestamped run directory."""
        base_path = Path(base_dir)
        # Get all directories matching timestamp pattern
        run_dirs = [d for d in base_path.iterdir() if d.is_dir() and len(d.name) == 19]
        if not run_dirs:
            raise ValueError(f"No run directories found in {base_dir}")
        # Sort by name (timestamp) and return latest
        return str(sorted(run_dirs)[-1])

    def load_experiment_data(self):
        """Load all runs from the phase diagram experiment."""
        all_runs = []
        print(f"Scanning directory: {self.data_dir}")
        run_dirs = list(self.data_dir.glob("*/"))
        print(f"Found {len(run_dirs)} run directories")
        
        # Navigate through Hydra's directory structure
        for run_dir in self.data_dir.glob("*/"):
            if run_dir.is_dir():
                metadata_file = run_dir / "metadata.yaml"
                data_file = run_dir / "simulation_data.npz"
                
                if metadata_file.exists() and data_file.exists():
                    # Load metadata to get parameters
                    from omegaconf import OmegaConf
                    metadata = OmegaConf.load(metadata_file)
                    
                    # Extract parameters from nested structure
                    params = metadata.get('params', {})
                    
                    # Set arena radius from first run
                    if self.arena_radius is None:
                        self.arena_radius = params.get('arena_radius', 50.0)
                    
                    # Load simulation data
                    data = np.load(data_file)
                    
                    # Data is stored as time series - get the final timestep
                    positions = data['positions']  # Shape: (n_timesteps, n_ants, 2)
                    behavioural_states = data['behavioural_states']  # Shape: (n_timesteps, n_ants)
                    
                    # Get final timestep
                    final_positions = positions[-1]
                    final_states = behavioural_states[-1]
                     
                    run_data = {
                        'threshold': params.get('pheromones', {}).get('pheromone_threshold'),
                        'steepness': params.get('pheromones', {}).get('pheromone_steepness'),
                        'seed': params.get('seed'),
                        'positions': final_positions,
                        'behavioural_states': final_states,
                        'time_series': {  # Keep full history for temporal analysis
                            'time': data['time'],
                            'positions': positions,
                            'behavioural_states': behavioural_states
                        }
                    }
                    all_runs.append(run_data)
     
        return pd.DataFrame(all_runs)
    
    def _parse_hydra_dirname(self, dirname):
        """Parse Hydra's override dirname to extract parameters."""
        params = {}
        # Split by comma and parse each override
        for override in dirname.split(','):
            if '=' in override:
                key, value = override.split('=')
                try:
                    params[key] = float(value)
                except ValueError:
                    params[key] = value
        return params
    
    def calculate_clustering_metrics(self, positions):
        """Calculate comprehensive clustering metrics for a single configuration."""
        n_ants = len(positions)
        
        # Use DBSCAN for cluster identification
        clustering = DBSCAN(eps=self.cluster_threshold, min_samples=2).fit(positions)
        labels = clustering.labels_
        
        # Number of clusters (excluding noise points labeled as -1)
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        
        # Cluster sizes
        cluster_sizes = []
        for cluster_id in set(labels):
            if cluster_id != -1:  # Exclude noise
                cluster_sizes.append(np.sum(labels == cluster_id))
        
        # Key metrics
        metrics = {
            'n_clusters': n_clusters,
            'n_isolated': np.sum(labels == -1),  # Ants not in any cluster
            'mean_cluster_size': np.mean(cluster_sizes) if cluster_sizes else 0,
            'max_cluster_size': max(cluster_sizes) if cluster_sizes else 0,
            'largest_cluster_fraction': max(cluster_sizes) / n_ants if cluster_sizes else 0,
            'clustering_coefficient': self._calculate_clustering_coefficient(positions),
            'spatial_entropy': self._calculate_spatial_entropy(positions),
            'mean_nn_distance': self._mean_nearest_neighbour_distance(positions)
        }
        
        return metrics
    
    def _calculate_clustering_coefficient(self, positions):
        """
        Calculate a global clustering coefficient.
        Fraction of ant pairs within threshold distance.
        """
        distances = pdist(positions)
        n_close_pairs = np.sum(distances < self.cluster_threshold)
        n_total_pairs = len(distances)
        return n_close_pairs / n_total_pairs if n_total_pairs > 0 else 0
    
    def _calculate_spatial_entropy(self, positions, grid_size=10):
        """Calculate Shannon entropy of spatial distribution."""
        # Create 2D histogram
        arena_radius = self.arena_radius
        bins = np.linspace(-arena_radius, arena_radius, grid_size + 1)
        H, _, _ = np.histogram2d(positions[:, 0], positions[:, 1], bins=[bins, bins])
        
        # Normalise and calculate entropy
        H_norm = H / np.sum(H)
        H_norm = H_norm[H_norm > 0]  # Remove zeros
        entropy = -np.sum(H_norm * np.log(H_norm))
        
        # Normalise by maximum entropy
        max_entropy = np.log(grid_size * grid_size)
        return entropy / max_entropy
    
    def _mean_nearest_neighbour_distance(self, positions):
        """Calculate mean nearest-neighbour distance."""
        distances = squareform(pdist(positions))
        np.fill_diagonal(distances, np.inf)
        nn_distances = np.min(distances, axis=1)
        return np.mean(nn_distances)
    
    def create_phase_diagram(self, df_results):
        """Create publication-quality phase diagrams."""
        # Aggregate metrics across seeds
        grouped = df_results.groupby(['threshold', 'steepness'])
        
        # Calculate means and standard errors
        metrics_to_plot = [
            'largest_cluster_fraction',
            'n_clusters', 
            'clustering_coefficient',
            'spatial_entropy'
        ]
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()
        
        for idx, metric in enumerate(metrics_to_plot):
            ax = axes[idx]
            
            # Pivot data for heatmap
            pivot_mean = grouped[metric].mean().reset_index().pivot(
                index='steepness', columns='threshold', values=metric
            )
            
            # Create heatmap
            im = ax.imshow(pivot_mean, aspect='auto', origin='lower', 
                          cmap='viridis' if metric != 'spatial_entropy' else 'viridis_r')
            
            # Set ticks and labels
            thresholds = sorted(df_results['threshold'].unique())
            steepnesses = sorted(df_results['steepness'].unique())
            
            ax.set_xticks(range(len(thresholds)))
            ax.set_xticklabels([f'{t:.1f}' for t in thresholds])
            ax.set_yticks(range(len(steepnesses)))
            ax.set_yticklabels([f'{s:.1f}' for s in steepnesses])
            
            ax.set_xlabel('Pheromone Threshold')
            ax.set_ylabel('Pheromone Steepness')
            
            # Add colorbar
            cbar = plt.colorbar(im, ax=ax)
            
            # Set title with nice formatting
            titles = {
                'largest_cluster_fraction': 'Largest Cluster Fraction',
                'n_clusters': 'Number of Clusters',
                'clustering_coefficient': 'Clustering Coefficient',
                'spatial_entropy': 'Spatial Entropy (normalised)'
            }
            ax.set_title(titles[metric])
            
            # Add phase boundary lines if clear transitions exist
            if metric == 'largest_cluster_fraction':
                self._add_phase_boundaries(ax, pivot_mean)
        
        plt.tight_layout()
        return fig

    def plot_sample_configurations(self, df, df_results):
        """Plot sample spatial configurations to visualize clustering."""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        # Select representative parameter combinations
        sample_params = [
            (1.0, 0.5),   # Low threshold, low steepness
            (5.0, 0.5),   # Medium threshold, low steepness
            (10.0, 0.5),  # High threshold, low steepness
            (1.0, 8.0),   # Low threshold, high steepness
            (5.0, 8.0),   # Medium threshold, high steepness
            (10.0, 8.0),  # High threshold, high steepness
        ]
        
        for idx, (thresh, steep) in enumerate(sample_params):
            ax = axes[idx]
            
            # Find a run with these parameters
            mask = (df['threshold'] == thresh) & (df['steepness'] == steep)
            if mask.any():
                run = df[mask].iloc[0]  # Take first seed
                positions = run['positions']
                
                # Plot positions
                ax.scatter(positions[:, 0], positions[:, 1], alpha=0.6, s=50)
                ax.set_xlim(-self.arena_radius, self.arena_radius)
                ax.set_ylim(-self.arena_radius, self.arena_radius)
                ax.set_aspect('equal')
                ax.set_title(f'T={thresh}, k={steep}')
                
                # Add circle for arena boundary
                circle = plt.Circle((0, 0), self.arena_radius, fill=False, linestyle='--', color='gray')
                ax.add_patch(circle)
        
        plt.tight_layout()
        return fig

    def _add_phase_boundaries(self, ax, data):
        """Identify and draw phase boundaries based on sharp transitions."""
        # Calculate gradient to find sharp transitions
        grad_x = np.gradient(data, axis=1)
        grad_y = np.gradient(data, axis=0)
        grad_mag = np.sqrt(grad_x**2 + grad_y**2)
        
        # Find high gradient regions (phase boundaries)
        threshold = np.percentile(grad_mag, 90)
        
        # Could add contour lines here if clear boundaries exist
        # ax.contour(grad_mag > threshold, levels=[0.5], colors='red', linewidths=2)
    
    def analyse_phase_transitions(self, df_results):
        """Detailed analysis of phase transitions."""
        # Group by parameters
        grouped = df_results.groupby(['threshold', 'steepness'])
        
        # Calculate order parameter (largest cluster fraction)
        order_param = grouped['largest_cluster_fraction'].agg(['mean', 'std', 'sem'])
        
        # Find critical threshold for each steepness
        critical_thresholds = []
        for steepness in sorted(df_results['steepness'].unique()):
            subset = order_param.xs(steepness, level='steepness')
            
            # Find threshold where order parameter crosses 0.5
            if len(subset) > 1:
                # Interpolate to find crossing point
                thresholds = subset.index
                values = subset['mean'].values
                
                # Find where it crosses 0.5
                idx = np.where(np.diff(np.sign(values - 0.5)))[0]
                if len(idx) > 0:
                    # Linear interpolation
                    t1, t2 = thresholds[idx[0]], thresholds[idx[0] + 1]
                    v1, v2 = values[idx[0]], values[idx[0] + 1]
                    critical_t = t1 + (t2 - t1) * (0.5 - v1) / (v2 - v1)
                    critical_thresholds.append((steepness, critical_t))
        
        return critical_thresholds
    
    def create_analysis_report(self, df_results):
        """Generate a comprehensive analysis report."""
        print("=== Phase Diagram Analysis Report ===\n")
        
        # Overall statistics
        print(f"Total runs analysed: {len(df_results)}")
        print(f"Parameter ranges:")
        print(f"  Threshold: {df_results['threshold'].min():.1f} - {df_results['threshold'].max():.1f}")
        print(f"  Steepness: {df_results['steepness'].min():.1f} - {df_results['steepness'].max():.1f}")
        print(f"  Seeds: {df_results['seed'].nunique()}\n")
        
        # Identify phases
        grouped = df_results.groupby(['threshold', 'steepness'])
        
        # Phase classification based on metrics
        for (threshold, steepness), group in grouped:
            mean_lcf = group['largest_cluster_fraction'].mean()
            mean_clusters = group['n_clusters'].mean()
            
            if mean_lcf < 0.2:
                phase = "Dispersed"
            elif mean_lcf > 0.8:
                phase = "Single Giant Cluster"
            elif mean_clusters > 5:
                phase = "Multiple Clusters"
            else:
                phase = "Transitional"
            
            if phase in ["Single Giant Cluster", "Transitional"]:
                print(f"T={threshold:.1f}, k={steepness:.1f}: {phase} (LCF={mean_lcf:.2f})")
        
        # Critical thresholds
        critical = self.analyse_phase_transitions(df_results)
        print("\nCritical thresholds (where LCF = 0.5):")
        for steepness, threshold in critical:
            print(f"  Steepness {steepness:.1f}: Critical threshold ≈ {threshold:.2f}")
        if len(critical) == 0:
            print("  No critical thresholds found - examining LCF distribution:")
            # Show the range of LCF values
            grouped = df_results.groupby(['threshold', 'steepness'])
            lcf_summary = grouped['largest_cluster_fraction'].mean()
            print(f"  LCF range: {lcf_summary.min():.3f} to {lcf_summary.max():.3f}")
            print("  This suggests the system may not exhibit clear phase transitions in this parameter range")

# Main analysis workflow
def main():
    # Load data
    latest_dir = PhaseAnalyser.find_latest_run("/data/michael/multirun/stigmergy/phase_diagram/")
    analyser = PhaseAnalyser(latest_dir)
    
    # Load all experimental runs
    print("Loading experimental data...")
    df = analyser.load_experiment_data()
    print(latest_dir)
    print(df.head())
    
    # Calculate metrics for each run
    print("Calculating clustering metrics...")
    all_metrics = []
    for idx, row in df.iterrows():
        metrics = analyser.calculate_clustering_metrics(row['positions'])
        metrics.update({
            'threshold': row['threshold'],
            'steepness': row['steepness'],
            'seed': row['seed']
        })
        all_metrics.append(metrics)
    
    df_results = pd.DataFrame(all_metrics)

    # Add debugging: print summary statistics
    print("\n=== Clustering Metrics Summary ===")
    print(df_results[['threshold', 'steepness', 'largest_cluster_fraction', 'n_clusters', 'clustering_coefficient']].describe())
    
    # Print a sample of the results grouped by parameters
    print("\n=== Sample Results by Parameters ===")
    grouped = df_results.groupby(['threshold', 'steepness'])
    sample_params = [(1.0, 0.5), (5.0, 4.0), (10.0, 16.0)]  # Low, medium, high
    for t, s in sample_params:
        if (t, s) in grouped.groups:
            group = grouped.get_group((t, s))
            print(f"\nThreshold={t}, Steepness={s}:")
            print(f"  Mean LCF: {group['largest_cluster_fraction'].mean():.3f} ± {group['largest_cluster_fraction'].std():.3f}")
            print(f"  Mean clusters: {group['n_clusters'].mean():.1f}")
            print(f"  Mean clustering coeff: {group['clustering_coefficient'].mean():.3f}")
    
    # Check behavioural states distribution
    print("\n=== Behavioural States Check ===")
    print(f"Sample behavioural states from last run: {df.iloc[-1]['behavioural_states'][:10]}")

    # Analyse behavioural states distribution
    print("\n=== Behavioural States Analysis ===")
    for idx in range(min(3, len(df))):
        states = df.iloc[idx]['behavioural_states']
        unique, counts = np.unique(states, return_counts=True)
        print(f"Run {idx} - States distribution:")
        for state, count in zip(unique, counts):
            print(f"  State {state}: {count} ants ({count/len(states)*100:.1f}%)")

    # Create a histogram of LCF values to see the distribution
    print("\n=== LCF Distribution ===")
    lcf_values = df_results['largest_cluster_fraction']
    print(f"Values > 0.1: {np.sum(lcf_values > 0.1)} runs")
    print(f"Values > 0.5: {np.sum(lcf_values > 0.5)} runs")
    print(f"Top 5 LCF values: {sorted(lcf_values, reverse=True)[:5]}")
    
    # Check clustering by threshold at fixed steepness
    print("\n=== Clustering vs Threshold (at steepness=4.0) ===")
    steep_4 = df_results[df_results['steepness'] == 4.0]
    if not steep_4.empty:
        thresh_grouped = steep_4.groupby('threshold')['largest_cluster_fraction'].agg(['mean', 'std'])
        print(thresh_grouped)
    
    # Also create the sample configurations plot
    fig_samples = analyser.plot_sample_configurations(df, df_results)
    fig_samples.savefig('sample_configurations_exp1.pdf', dpi=300, bbox_inches='tight')
    
    
    # Check if positions are changing over time
    print("\n=== Temporal Analysis (first run) ===")
    time_series = df.iloc[0]['time_series']
    positions_t0 = time_series['positions'][0]
    positions_tf = time_series['positions'][-1]
    movement = np.linalg.norm(positions_tf - positions_t0, axis=1)
    print(f"Mean displacement: {movement.mean():.2f}")
    print(f"Max displacement: {movement.max():.2f}")
    print(f"Stationary ants (moved < 1.0): {np.sum(movement < 1.0)}")
    
    # Find the run with highest LCF
    max_lcf_idx = df_results['largest_cluster_fraction'].idxmax()
    max_lcf_row = df_results.iloc[max_lcf_idx]
    print(f"\n=== Run with highest clustering ===")
    print(f"Parameters: T={max_lcf_row['threshold']}, k={max_lcf_row['steepness']}, seed={max_lcf_row['seed']}")
    print(f"LCF: {max_lcf_row['largest_cluster_fraction']:.3f}, n_clusters: {max_lcf_row['n_clusters']}")
    

    # Generate phase diagram
    print("Creating phase diagrams...")
    fig = analyser.create_phase_diagram(df_results)
    fig.savefig('phase_diagram_exp1.pdf', dpi=300, bbox_inches='tight')
    
    # Generate analysis report
    analyser.create_analysis_report(df_results)
    
    # Save processed data
    df_results.to_csv('phase_diagram_metrics.csv', index=False)
    
    return df_results

if __name__ == "__main__":
    df_results = main()