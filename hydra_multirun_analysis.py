import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import seaborn as sns
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import DBSCAN
from scipy.stats import sem
import matplotlib.patches as patches
from matplotlib.colors import LogNorm
from scipy.ndimage import gaussian_filter

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
    
    def create_enhanced_phase_diagram(self, df_results):
        """Create enhanced phase diagrams with better visualisation."""
        # Aggregate metrics across seeds
        grouped = df_results.groupby(['threshold', 'steepness'])
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        # 1. Main phase diagram - Largest Cluster Fraction
        ax = axes[0]
        pivot_lcf = grouped['largest_cluster_fraction'].mean().reset_index().pivot(
            index='steepness', columns='threshold', values='largest_cluster_fraction'
        )
        
        # Use better colormap and add contours
        im = ax.imshow(pivot_lcf, aspect='auto', origin='lower', 
                      cmap='RdYlBu_r', vmin=0, vmax=1)
        
        # Add contour lines at phase boundaries
        contours = ax.contour(pivot_lcf, levels=[0.1, 0.5, 0.9], 
                             colors='black', linewidths=1.5, alpha=0.7)
        ax.clabel(contours, inline=True, fontsize=8)
        
        # Set labels
        thresholds = sorted(df_results['threshold'].unique())
        steepnesses = sorted(df_results['steepness'].unique())
        ax.set_xticks(range(len(thresholds)))
        ax.set_xticklabels([f'{t:.0f}' for t in thresholds])
        ax.set_yticks(range(len(steepnesses)))
        ax.set_yticklabels([f'{s:.1f}' for s in steepnesses])
        ax.set_xlabel('Pheromone Threshold')
        ax.set_ylabel('Pheromone Steepness')
        ax.set_title('Largest Cluster Fraction (with phase boundaries)')
        plt.colorbar(im, ax=ax)
        
        # 2. Standard deviation plot
        ax = axes[1]
        pivot_std = grouped['largest_cluster_fraction'].std().reset_index().pivot(
            index='steepness', columns='threshold', values='largest_cluster_fraction'
        )
        im = ax.imshow(pivot_std, aspect='auto', origin='lower', cmap='viridis')
        ax.set_xticks(range(len(thresholds)))
        ax.set_xticklabels([f'{t:.0f}' for t in thresholds])
        ax.set_yticks(range(len(steepnesses)))
        ax.set_yticklabels([f'{s:.1f}' for s in steepnesses])
        ax.set_xlabel('Pheromone Threshold')
        ax.set_ylabel('Pheromone Steepness')
        ax.set_title('LCF Standard Deviation (phase transition regions)')
        plt.colorbar(im, ax=ax)
        
        # 3. Number of clusters
        ax = axes[2]
        pivot_clusters = grouped['n_clusters'].mean().reset_index().pivot(
            index='steepness', columns='threshold', values='n_clusters'
        )
        im = ax.imshow(pivot_clusters, aspect='auto', origin='lower', cmap='viridis')
        ax.set_xticks(range(len(thresholds)))
        ax.set_xticklabels([f'{t:.0f}' for t in thresholds])
        ax.set_yticks(range(len(steepnesses)))
        ax.set_yticklabels([f'{s:.1f}' for s in steepnesses])
        ax.set_xlabel('Pheromone Threshold')
        ax.set_ylabel('Pheromone Steepness')
        ax.set_title('Mean Number of Clusters')
        plt.colorbar(im, ax=ax)
        
        # 4. Phase classification
        ax = axes[3]
        phase_map = np.zeros_like(pivot_lcf)
        for i in range(pivot_lcf.shape[0]):
            for j in range(pivot_lcf.shape[1]):
                lcf = pivot_lcf.iloc[i, j]
                n_clust = pivot_clusters.iloc[i, j]
                if lcf > 0.6:
                    phase_map[i, j] = 3  # Single giant cluster
                elif lcf > 0.3:
                    phase_map[i, j] = 2  # Transitional
                elif n_clust > 10:
                    phase_map[i, j] = 1  # Multiple small clusters
                else:
                    phase_map[i, j] = 0  # Dispersed
        
        cmap = plt.cm.colors.ListedColormap(['darkblue', 'blue', 'yellow', 'red'])
        im = ax.imshow(phase_map, aspect='auto', origin='lower', cmap=cmap)
        ax.set_xticks(range(len(thresholds)))
        ax.set_xticklabels([f'{t:.0f}' for t in thresholds])
        ax.set_yticks(range(len(steepnesses)))
        ax.set_yticklabels([f'{s:.1f}' for s in steepnesses])
        ax.set_xlabel('Pheromone Threshold')
        ax.set_ylabel('Pheromone Steepness')
        ax.set_title('Phase Classification')
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='darkblue', label='Dispersed'),
            Patch(facecolor='blue', label='Multiple clusters'),
            Patch(facecolor='yellow', label='Transitional'),
            Patch(facecolor='red', label='Giant cluster')
        ]
        ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.3, 1))
        
        # 5. Clustering coefficient
        ax = axes[4]
        pivot_cc = grouped['clustering_coefficient'].mean().reset_index().pivot(
            index='steepness', columns='threshold', values='clustering_coefficient'
        )
        im = ax.imshow(pivot_cc, aspect='auto', origin='lower', cmap='viridis')
        ax.set_xticks(range(len(thresholds)))
        ax.set_xticklabels([f'{t:.0f}' for t in thresholds])
        ax.set_yticks(range(len(steepnesses)))
        ax.set_yticklabels([f'{s:.1f}' for s in steepnesses])
        ax.set_xlabel('Pheromone Threshold')
        ax.set_ylabel('Pheromone Steepness')
        ax.set_title('Clustering Coefficient')
        plt.colorbar(im, ax=ax)
        
        # 6. Line plots showing transitions
        ax = axes[5]
        # Plot LCF vs threshold for different steepness values
        for s in [0.5, 1.0, 2.0, 4.0, 8.0]:
            subset = df_results[df_results['steepness'] == s]
            if not subset.empty:
                thresh_grouped = subset.groupby('threshold')['largest_cluster_fraction']
                means = thresh_grouped.mean()
                stds = thresh_grouped.std()
                ax.errorbar(means.index, means.values, yerr=stds.values, 
                           label=f'k={s}', marker='o', capsize=3)
        
        ax.set_xlabel('Pheromone Threshold')
        ax.set_ylabel('Largest Cluster Fraction')
        ax.set_title('Phase Transitions for Different Steepness Values')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-0.1, 1.1)
        
        plt.tight_layout()
        return fig
    
    def plot_sample_configurations(self, df, df_results):
        """Plot sample spatial configurations to visualize clustering."""
        fig, axes = plt.subplots(3, 3, figsize=(15, 15))
        axes = axes.flatten()
        
        # Find configurations with different clustering levels
        # Sort by LCF to get a range
        df_with_metrics = df.merge(
            df_results[['threshold', 'steepness', 'seed', 'largest_cluster_fraction']], 
            on=['threshold', 'steepness', 'seed']
        )
        df_sorted = df_with_metrics.sort_values('largest_cluster_fraction')
        
        # Select 9 evenly spaced configurations
        indices = np.linspace(0, len(df_sorted)-1, 9, dtype=int)
        
        for idx, ax_idx in enumerate(indices):
            ax = axes[idx]
            run = df_sorted.iloc[ax_idx]
            positions = run['positions']
            states = run['behavioural_states']
            
            # Color by behavioural state
            colors = ['red' if s == 0 else 'blue' if s == 1 else 'green' for s in states]
            
            # Plot positions
            ax.scatter(positions[:, 0], positions[:, 1], c=colors, alpha=0.6, s=50)
            ax.set_xlim(-self.arena_radius, self.arena_radius)
            ax.set_ylim(-self.arena_radius, self.arena_radius)
            ax.set_aspect('equal')
            
            # Add title with parameters and LCF
            ax.set_title(f'T={run["threshold"]:.1f}, k={run["steepness"]:.1f}\n' + 
                        f'LCF={run["largest_cluster_fraction"]:.3f}')
            
            # Add circle for arena boundary
            circle = plt.Circle((0, 0), self.arena_radius, fill=False, 
                              linestyle='--', color='gray', alpha=0.5)
            ax.add_patch(circle)
            
            # Remove tick labels for cleaner look
            ax.set_xticks([])
            ax.set_yticks([])
        
        # Add legend
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor='red', 
                   markersize=10, label='Returning (0)'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', 
                   markersize=10, label='Exploring (1)'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='green', 
                   markersize=10, label='Arrested (2)')
        ]
        fig.legend(handles=legend_elements, loc='upper center', 
                  bbox_to_anchor=(0.5, 0.98), ncol=3)
        
        plt.tight_layout()
        return fig
    
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
        print("\n=== Phase Diagram Analysis Report ===\n")
        
        # Overall statistics
        print(f"Total runs analysed: {len(df_results)}")
        print(f"Parameter ranges:")
        print(f"  Threshold: {df_results['threshold'].min():.1f} - {df_results['threshold'].max():.1f}")
        print(f"  Steepness: {df_results['steepness'].min():.1f} - {df_results['steepness'].max():.1f}")
        print(f"  Seeds: {df_results['seed'].nunique()}\n")
        
        # Identify phases
        grouped = df_results.groupby(['threshold', 'steepness'])
        
        print("Key findings:")
        print("-" * 40)
        
        # Find parameter regions for each phase
        dispersed = []
        transitional = []
        clustered = []
        
        for (threshold, steepness), group in grouped:
            mean_lcf = group['largest_cluster_fraction'].mean()
            mean_clusters = group['n_clusters'].mean()
            
            if mean_lcf > 0.6:
                clustered.append((threshold, steepness, mean_lcf))
            elif mean_lcf > 0.3:
                transitional.append((threshold, steepness, mean_lcf))
            elif mean_lcf < 0.1:
                dispersed.append((threshold, steepness, mean_lcf))
        
        print(f"\nHighly clustered phase (LCF > 0.6): {len(clustered)} parameter combinations")
        if clustered:
            print("  Examples:")
            for t, s, lcf in clustered[:3]:
                print(f"    T={t:.1f}, k={s:.1f}: LCF={lcf:.3f}")
        
        print(f"\nTransitional phase (0.3 < LCF < 0.6): {len(transitional)} parameter combinations")
        if transitional:
            print("  Examples:")
            for t, s, lcf in transitional[:3]:
                print(f"    T={t:.1f}, k={s:.1f}: LCF={lcf:.3f}")
        
        print(f"\nDispersed phase (LCF < 0.1): {len(dispersed)} parameter combinations")
        
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

# Main analysis workflow
def main():
    # Load data
    latest_dir = PhaseAnalyser.find_latest_run("/data/michael/multirun/stigmergy/phase_diagram/")
    analyser = PhaseAnalyser(latest_dir)
    
    # Load all experimental runs
    print("Loading experimental data...")
    df = analyser.load_experiment_data()
    print(f"Loaded {len(df)} runs from {latest_dir}")
    
    # Calculate metrics for each run
    print("\nCalculating clustering metrics...")
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
    
    # Generate enhanced phase diagram
    print("\nCreating enhanced phase diagrams...")
    fig = analyser.create_enhanced_phase_diagram(df_results)
    fig.savefig('phase_diagram_enhanced.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create sample configuration plots
    print("Creating sample configuration plots...")
    fig_samples = analyser.plot_sample_configurations(df, df_results)
    fig_samples.savefig('sample_configurations.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Generate analysis report
    analyser.create_analysis_report(df_results)
    
    # Save processed data
    df_results.to_csv('phase_diagram_metrics.csv', index=False)
    print(f"\nSaved metrics to phase_diagram_metrics.csv")
    
    return df_results

if __name__ == "__main__":
    df_results = main()