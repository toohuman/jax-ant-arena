import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import DBSCAN
from omegaconf import OmegaConf

# Set publication-quality defaults
plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10

class ArrestedStateAnalyser:
    def __init__(self, data_dir, ant_length=1.0):
        self.data_dir = Path(data_dir)
        self.ant_length = ant_length
        self.cluster_threshold = 2.5 * ant_length
        self.arena_radius = None

    @staticmethod
    def find_latest_run(base_dir):
        """Find the latest timestamped run directory."""
        base_path = Path(base_dir)
        run_dirs = [d for d in base_path.iterdir() if d.is_dir() and len(d.name) == 19]
        if not run_dirs:
            raise ValueError(f"No run directories found in {base_dir}")
        return str(sorted(run_dirs)[-1])

    def load_experiment_data(self):
        """Load runs focusing on temporal data."""
        all_runs = []
        print(f"Scanning directory: {self.data_dir}")
        
        for run_dir in self.data_dir.glob("*/"):
            if run_dir.is_dir():
                metadata_file = run_dir / "metadata.yaml"
                data_file = run_dir / "simulation_data.npz"
                
                if metadata_file.exists() and data_file.exists():
                    metadata = OmegaConf.load(metadata_file)
                    params = metadata.get('params', {})
                    
                    if self.arena_radius is None:
                        self.arena_radius = params.get('arena_radius', 50.0)
                    
                    data = np.load(data_file)
                    
                    run_data = {
                        'threshold': params.get('pheromones', {}).get('pheromone_threshold'),
                        'steepness': params.get('pheromones', {}).get('pheromone_steepness'),
                        'seed': params.get('seed'),
                        'time': data['time'],
                        'positions': data['positions'],
                        'behavioural_states': data['behavioural_states']
                    }
                    all_runs.append(run_data)
        
        return pd.DataFrame(all_runs)
    
    def calculate_temporal_metrics_with_states(self, positions_series, states_series, time_points):
        """Calculate clustering metrics and state fractions over time."""
        temporal_metrics = {
            'time': time_points,
            'largest_cluster_fraction': [],
            'n_clusters': [],
            'clustering_coefficient': [],
            'arrested_fraction': [],
            'exploring_fraction': [],
            'returning_fraction': []
        }
        
        for t_idx in range(len(time_points)):
            positions = positions_series[t_idx]
            states = states_series[t_idx]
            
            # Clustering metrics
            clustering = DBSCAN(eps=self.cluster_threshold, min_samples=3).fit(positions)
            labels = clustering.labels_
            
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            cluster_sizes = []
            for cluster_id in set(labels):
                if cluster_id != -1:
                    cluster_sizes.append(np.sum(labels == cluster_id))
            
            # Calculate metrics
            temporal_metrics['n_clusters'].append(n_clusters)
            temporal_metrics['largest_cluster_fraction'].append(
                max(cluster_sizes) / len(positions) if cluster_sizes else 0
            )
            
            # Clustering coefficient
            distances = pdist(positions)
            n_close = np.sum(distances < self.cluster_threshold)
            n_total = len(distances) if len(distances) > 0 else 1
            temporal_metrics['clustering_coefficient'].append(n_close / n_total)
             
            # Calculate arrested-only clustering metrics
            arrested_mask = (states == 2)
            arrested_positions = positions[arrested_mask]
            
            if len(arrested_positions) >= 2:
                # Cluster only arrested ants
                clustering_arrested = DBSCAN(eps=self.cluster_threshold, min_samples=2).fit(arrested_positions)
                labels_arrested = clustering_arrested.labels_
                
                n_clusters_arrested = len(set(labels_arrested)) - (1 if -1 in labels_arrested else 0)
                cluster_sizes_arrested = []
                for cluster_id in set(labels_arrested):
                    if cluster_id != -1:
                        cluster_sizes_arrested.append(np.sum(labels_arrested == cluster_id))
                
                # Arrested-only metrics
                temporal_metrics['n_clusters_arrested'] = n_clusters_arrested
                temporal_metrics['arrested_lcf'] = (
                    max(cluster_sizes_arrested) / len(arrested_positions) 
                    if cluster_sizes_arrested else 0
                )
                temporal_metrics['arrested_in_clusters'] = (
                    sum(cluster_sizes_arrested) / len(positions)  # Fraction of ALL ants
                )
            else:
                temporal_metrics['n_clusters_arrested'] = 0
                temporal_metrics['arrested_lcf'] = 0
                temporal_metrics['arrested_in_clusters'] = 0
            
            # State fractions
            n_ants = len(states)
            temporal_metrics['returning_fraction'].append(np.sum(states == 0) / n_ants)
            temporal_metrics['exploring_fraction'].append(np.sum(states == 1) / n_ants)
            temporal_metrics['arrested_fraction'].append(np.sum(states == 2) / n_ants)
        
        return temporal_metrics
    
    def plot_temporal_dynamics_with_states(self, df, target_params=None, n_examples=6):
        """Plot temporal dynamics including arrested state fraction."""
        
        if target_params is None:
            # Select interesting parameter combinations
            target_params = [
                (2.0, 4.0),   # Your sweet spot
                (2.0, 2.0),   # Lower steepness
                (2.0, 8.0),   # Higher steepness
                (1.5, 4.0),   # Lower threshold
                (2.5, 4.0),   # Higher threshold
                (3.0, 4.0),   # Even higher threshold
            ]
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        for idx, (thresh, steep) in enumerate(target_params[:n_examples]):
            ax = axes[idx]
            
            # Find a run with these parameters
            mask = (df['threshold'] == thresh) & (df['steepness'] == steep)
            if not mask.any():
                # Find closest parameters
                distances = np.sqrt((df['threshold'] - thresh)**2 + 
                                  (df['steepness'] - steep)**2)
                closest_idx = distances.idxmin()
                run = df.iloc[closest_idx]
                actual_t = run['threshold']
                actual_k = run['steepness']
                print(f"No exact match for T={thresh}, k={steep}. Using T={actual_t}, k={actual_k}")
            else:
                run = df[mask].iloc[0]  # Take first seed
                actual_t = thresh
                actual_k = steep
            
            # Calculate temporal metrics
            temporal = self.calculate_temporal_metrics_with_states(
                run['positions'], run['behavioural_states'], run['time']
            )
            
            # Create plot with three y-axes
            # Primary axis: LCF and arrested fraction
            line1 = ax.plot(temporal['time'], temporal['largest_cluster_fraction'], 
                           'b-', linewidth=2, label='LCF')
            line2 = ax.plot(temporal['time'], temporal['arrested_fraction'], 
                           'g-', linewidth=2, label='Arrested fraction')
            
            ax.set_xlabel('Time')
            ax.set_ylabel('Fraction', color='black')
            ax.set_ylim(0, 1)
            
            # Secondary axis: Number of clusters
            ax2 = ax.twinx()
            line3 = ax2.plot(temporal['time'], temporal['n_clusters'], 
                            'r--', linewidth=1.5, label='N clusters')
            ax2.set_ylabel('Number of Clusters', color='r')
            ax2.tick_params(axis='y', labelcolor='r')
            
            # Title and legend
            ax.set_title(f'T={actual_t:.1f}, k={actual_k:.1f}')
            
            # Combine legends
            lines = line1 + line2 + line3
            labels = [l.get_label() for l in lines]
            ax.legend(lines, labels, loc='center right')
            
            ax.grid(True, alpha=0.3)
        
        plt.suptitle('Temporal Dynamics: Clustering and Arrested States', fontsize=16)
        plt.tight_layout()
        return fig
    
    def plot_state_correlation_analysis(self, df):
        """Analyse correlation between arrested fraction and clustering."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        
        # Collect data across all runs
        all_arrested = []
        all_lcf = []
        all_params = []
        
        for idx, run in df.iterrows():
            temporal = self.calculate_temporal_metrics_with_states(
                run['positions'], run['behavioural_states'], run['time']
            )
            
            # Sample every 10th timepoint to avoid overcrowding
            sample_indices = range(0, len(temporal['time']), 10)
            for i in sample_indices:
                all_arrested.append(temporal['arrested_fraction'][i])
                all_lcf.append(temporal['largest_cluster_fraction'][i])
                all_params.append((run['threshold'], run['steepness']))
        
        all_arrested = np.array(all_arrested)
        all_lcf = np.array(all_lcf)
        
        # 1. Scatter plot: Arrested fraction vs LCF
        ax = axes[0, 0]
        scatter = ax.scatter(all_arrested, all_lcf, alpha=0.3, s=20, c=all_arrested, cmap='viridis')
        ax.set_xlabel('Arrested Fraction')
        ax.set_ylabel('Largest Cluster Fraction')
        ax.set_title('Correlation: Arrested State vs Clustering')
        ax.grid(True, alpha=0.3)
        
        # Add correlation coefficient
        corr = np.corrcoef(all_arrested, all_lcf)[0, 1]
        ax.text(0.05, 0.95, f'Correlation: {corr:.3f}', 
                transform=ax.transAxes, verticalalignment='top')
        
        # 2. Time delay analysis
        ax = axes[0, 1]
        # For a few representative runs, plot arrested fraction vs LCF with time
        representative_params = [(2.0, 4.0), (2.0, 8.0), (3.0, 4.0)]
        
        for thresh, steep in representative_params:
            mask = (df['threshold'] == thresh) & (df['steepness'] == steep)
            if mask.any():
                run = df[mask].iloc[0]
                temporal = self.calculate_temporal_metrics_with_states(
                    run['positions'], run['behavioural_states'], run['time']
                )
                
                # Plot with arrested fraction on x-axis, LCF on y-axis
                # Color represents time
                times = temporal['time']
                colors = plt.cm.plasma(times / times[-1])
                
                for i in range(1, len(times), 50):  # Sample to reduce points
                    ax.plot(temporal['arrested_fraction'][i-1:i+1], 
                           temporal['largest_cluster_fraction'][i-1:i+1],
                           color=colors[i], alpha=0.7, linewidth=2)
                
                # Mark start and end
                ax.scatter(temporal['arrested_fraction'][0], 
                          temporal['largest_cluster_fraction'][0],
                          marker='o', s=100, edgecolor='black', 
                          facecolor='green', label=f'Start T={thresh},k={steep}')
                ax.scatter(temporal['arrested_fraction'][-1], 
                          temporal['largest_cluster_fraction'][-1],
                          marker='s', s=100, edgecolor='black', 
                          facecolor='red')
        
        ax.set_xlabel('Arrested Fraction')
        ax.set_ylabel('Largest Cluster Fraction')
        ax.set_title('Temporal Evolution (color = time)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 3. Phase diagram: Peak arrested fraction
        ax = axes[1, 0]
        # Calculate peak arrested fraction for each parameter combination
        param_groups = {}
        for idx, run in df.iterrows():
            temporal = self.calculate_temporal_metrics_with_states(
                run['positions'], run['behavioural_states'], run['time']
            )
            key = (run['threshold'], run['steepness'])
            peak_arrested = max(temporal['arrested_fraction'])
            
            if key not in param_groups:
                param_groups[key] = []
            param_groups[key].append(peak_arrested)
        
        # Create grid
        thresholds = sorted(df['threshold'].unique())
        steepnesses = sorted(df['steepness'].unique())
        peak_grid = np.zeros((len(steepnesses), len(thresholds)))
        
        for (t, s), values in param_groups.items():
            t_idx = thresholds.index(t)
            s_idx = steepnesses.index(s)
            peak_grid[s_idx, t_idx] = np.mean(values)
        
        im = ax.imshow(peak_grid, aspect='auto', origin='lower', cmap='hot')
        ax.set_xticks(range(len(thresholds)))
        ax.set_xticklabels([f'{t:.0f}' for t in thresholds])
        ax.set_yticks(range(len(steepnesses)))
        ax.set_yticklabels([f'{s:.1f}' for s in steepnesses])
        ax.set_xlabel('Pheromone Threshold')
        ax.set_ylabel('Pheromone Steepness')
        ax.set_title('Peak Arrested Fraction')
        plt.colorbar(im, ax=ax)
        
        # 4. Time to peak arrest vs time to peak clustering
        ax = axes[1, 1]
        time_to_peak_arrest = []
        time_to_peak_cluster = []
        
        for idx, run in df.iterrows():
            temporal = self.calculate_temporal_metrics_with_states(
                run['positions'], run['behavioural_states'], run['time']
            )
            
            # Find time to peak
            peak_arrest_idx = np.argmax(temporal['arrested_fraction'])
            peak_cluster_idx = np.argmax(temporal['largest_cluster_fraction'])
            
            time_to_peak_arrest.append(temporal['time'][peak_arrest_idx])
            time_to_peak_cluster.append(temporal['time'][peak_cluster_idx])
        
        ax.scatter(time_to_peak_arrest, time_to_peak_cluster, alpha=0.5)
        ax.plot([0, max(time_to_peak_arrest)], [0, max(time_to_peak_arrest)], 
                'k--', label='y=x')
        ax.set_xlabel('Time to Peak Arrested Fraction')
        ax.set_ylabel('Time to Peak Clustering')
        ax.set_title('Timing of Arrest vs Clustering')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def create_summary_report(self, df):
        """Generate summary statistics about arrested states and clustering."""
        print("\n=== Arrested State and Clustering Analysis ===\n")
        
        # Analyse each parameter combination
        param_results = []
        
        for (thresh, steep), group in df.groupby(['threshold', 'steepness']):
            arrested_peaks = []
            lcf_peaks = []
            arrest_cluster_corrs = []
            
            for idx, run in group.iterrows():
                temporal = self.calculate_temporal_metrics_with_states(
                    run['positions'], run['behavioural_states'], run['time']
                )
                
                arrested = np.array(temporal['arrested_fraction'])
                lcf = np.array(temporal['largest_cluster_fraction'])
                
                arrested_peaks.append(np.max(arrested))
                lcf_peaks.append(np.max(lcf))
                
                # Correlation between arrested fraction and LCF
                if len(arrested) > 10:  # Need enough points
                    corr = np.corrcoef(arrested, lcf)[0, 1]
                    arrest_cluster_corrs.append(corr)
            
            param_results.append({
                'threshold': thresh,
                'steepness': steep,
                'mean_peak_arrested': np.mean(arrested_peaks),
                'mean_peak_lcf': np.mean(lcf_peaks),
                'mean_correlation': np.mean(arrest_cluster_corrs) if arrest_cluster_corrs else np.nan,
                'n_runs': len(group)
            })
        
        results_df = pd.DataFrame(param_results)
        
        # Find best parameters for emergent behaviour
        # High correlation + high peaks = strong emergent clustering
        results_df['emergence_score'] = (
            results_df['mean_correlation'] * 
            results_df['mean_peak_arrested'] * 
            results_df['mean_peak_lcf']
        )
        
        print("Top 5 parameter combinations for emergent clustering:")
        print("(High correlation between arrest and clustering)")
        print(results_df.nlargest(5, 'emergence_score')[
            ['threshold', 'steepness', 'mean_peak_arrested', 
             'mean_peak_lcf', 'mean_correlation', 'emergence_score']
        ].to_string())
        
        return results_df

# Main analysis
def main():
    # Load data
    latest_dir = ArrestedStateAnalyser.find_latest_run(
        "/data/michael/ants/multirun/stigmergy/phase_diagram/"
    )
    analyser = ArrestedStateAnalyser(latest_dir)
    
    print("Loading experimental data...")
    df = analyser.load_experiment_data()
    print(f"Loaded {len(df)} runs")
    
    # Create temporal dynamics plots with arrested states
    print("\nCreating temporal dynamics plots with arrested states...")
    fig = analyser.plot_temporal_dynamics_with_states(df)
    fig.savefig('temporal_dynamics_with_arrested.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create correlation analysis
    print("Analysing correlation between arrested states and clustering...")
    fig = analyser.plot_state_correlation_analysis(df)
    fig.savefig('arrested_clustering_correlation.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Generate summary report
    results_df = analyser.create_summary_report(df)
    results_df.to_csv('arrested_clustering_summary.csv', index=False)
    
    print("\nAnalysis complete! Check the PDF outputs.")
    
    return df, results_df

if __name__ == "__main__":
    df, results_df = main()