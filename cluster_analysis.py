import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from sklearn.cluster import DBSCAN
from omegaconf import OmegaConf

# Publication quality
plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['font.family'] = 'sans-serif'

class ArrestedClusteringAnalyser:
    def __init__(self, data_dir, ant_length=1.0):
        self.data_dir = Path(data_dir)
        self.ant_length = ant_length
        self.cluster_threshold = 2.5 * ant_length
        self.arena_radius = 50.0

    @staticmethod
    def find_latest_run(base_dir):
        base_path = Path(base_dir)
        run_dirs = [d for d in base_path.iterdir() if d.is_dir() and len(d.name) == 19]
        return str(sorted(run_dirs)[-1])

    def load_runs(self, target_params):
        """Load only runs matching target parameters."""
        runs = []
        for run_dir in self.data_dir.glob("*/"):
            if run_dir.is_dir():
                metadata_file = run_dir / "metadata.yaml"
                data_file = run_dir / "simulation_data.npz"
                
                if metadata_file.exists() and data_file.exists():
                    metadata = OmegaConf.load(metadata_file)
                    params = metadata.get('params', {})
                    
                    thresh = params.get('pheromones', {}).get('pheromone_threshold')
                    steep = params.get('pheromones', {}).get('pheromone_steepness')
                    
                    # Check if matches any target params
                    for t_thresh, t_steep in target_params:
                        if thresh == t_thresh and steep == t_steep:
                            data = np.load(data_file)
                            runs.append({
                                'threshold': thresh,
                                'steepness': steep,
                                'seed': params.get('seed'),
                                'time': data['time'],
                                'positions': data['positions'],
                                'states': data['behavioural_states']
                            })
                            break
        return runs

    def calculate_arrested_metrics(self, positions, states):
        """Calculate clustering metrics for arrested ants only."""
        arrested_mask = (states == 2)
        arrested_positions = positions[arrested_mask]
        n_arrested = len(arrested_positions)
        n_total = len(positions)
        
        if n_arrested < 2:
            return {
                'arrested_fraction': n_arrested / n_total,
                'arrested_lcf': 0,
                'clustered_fraction': 0,
                'n_arrested_clusters': 0
            }
        
        # Cluster arrested ants
        clustering = DBSCAN(eps=self.cluster_threshold, min_samples=2).fit(arrested_positions)
        labels = clustering.labels_
        
        # Calculate metrics
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        cluster_sizes = [np.sum(labels == i) for i in set(labels) if i != -1]
        
        return {
            'arrested_fraction': n_arrested / n_total,
            'arrested_lcf': max(cluster_sizes) / n_arrested if cluster_sizes else 0,
            'clustered_fraction': sum(cluster_sizes) / n_total,  # Fraction of ALL ants
            'n_arrested_clusters': n_clusters
        }

    def analyse_runs(self, runs):
        """Analyse temporal dynamics for all runs."""
        results = []
        
        for run in runs:
            # Sample time points to reduce computation
            sample_indices = range(0, len(run['time']), 10)  # Every 10th timestep
            
            temporal_data = {
                'time': [], 'arrested_fraction': [], 'arrested_lcf': [],
                'clustered_fraction': [], 'n_arrested_clusters': []
            }
            
            for idx in sample_indices:
                metrics = self.calculate_arrested_metrics(
                    run['positions'][idx], run['states'][idx]
                )
                temporal_data['time'].append(run['time'][idx])
                for key, value in metrics.items():
                    temporal_data[key].append(value)
            
            # Calculate summary statistics
            results.append({
                'threshold': run['threshold'],
                'steepness': run['steepness'],
                'seed': run['seed'],
                'max_arrested': max(temporal_data['arrested_fraction']),
                'max_lcf': max(temporal_data['arrested_lcf']),
                'max_clustered': max(temporal_data['clustered_fraction']),
                'final_clustered': temporal_data['clustered_fraction'][-1],
                'temporal_data': temporal_data
            })
        
        return pd.DataFrame(results)

    def find_optimal_params(self, df):
        """Identify optimal parameters based on emergent clustering."""
        # Group by parameters
        grouped = df.groupby(['threshold', 'steepness']).agg({
            'max_clustered': ['mean', 'std'],
            'final_clustered': ['mean', 'std'],
            'max_arrested': 'mean',
            'max_lcf': 'mean'
        }).round(3)
        
        # Score: high clustering, low variance, sustained over time
        grouped['score'] = (
            grouped['max_clustered']['mean'] * 
            grouped['final_clustered']['mean'] / 
            (grouped['max_clustered']['std'] + 0.01)
        )
        
        return grouped.sort_values('score', ascending=False)

    def create_figure(self, df, optimal_params):
        """Create single comprehensive figure."""
        fig = plt.figure(figsize=(14, 8))
        
        # Get optimal runs
        opt_thresh, opt_steep = optimal_params
        optimal_runs = df[(df['threshold'] == opt_thresh) & (df['steepness'] == opt_steep)]
        
        # 1. Temporal dynamics for optimal parameters (3 seeds)
        ax1 = plt.subplot(2, 3, (1, 4))
        for _, run in optimal_runs.head(3).iterrows():
            data = run['temporal_data']
            ax1.plot(data['time'], data['clustered_fraction'], 
                    alpha=0.8, label=f'Seed {int(run["seed"])}')
        
        ax1.set_xlabel('Time')
        ax1.set_ylabel('Fraction of Ants in Arrested Clusters')
        ax1.set_title(f'Optimal: T={opt_thresh}, k={opt_steep}')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, 1)
        
        # 2. Phase diagram: Max clustered fraction
        ax2 = plt.subplot(2, 3, 2)
        pivot = df.groupby(['threshold', 'steepness'])['max_clustered'].mean().reset_index()
        pivot_matrix = pivot.pivot(index='steepness', columns='threshold', values='max_clustered')
        
        im = ax2.imshow(pivot_matrix, aspect='auto', origin='lower', cmap='hot', vmin=0, vmax=1)
        ax2.set_title('Max Fraction in Arrested Clusters')
        
        # Mark optimal
        thresh_idx = list(pivot_matrix.columns).index(opt_thresh)
        steep_idx = list(pivot_matrix.index).index(opt_steep)
        ax2.plot(thresh_idx, steep_idx, 'c*', markersize=20, markeredgecolor='white', markeredgewidth=2)
        
        ax2.set_xticks(range(len(pivot_matrix.columns)))
        ax2.set_xticklabels([f'{t:.0f}' for t in pivot_matrix.columns])
        ax2.set_yticks(range(len(pivot_matrix.index)))
        ax2.set_yticklabels([f'{s:.1f}' for s in pivot_matrix.index])
        ax2.set_xlabel('Threshold')
        ax2.set_ylabel('Steepness')
        plt.colorbar(im, ax=ax2)
        
        # 3. Arrested vs Clustered correlation
        ax3 = plt.subplot(2, 3, 3)
        for _, run in optimal_runs.iterrows():
            data = run['temporal_data']
            ax3.scatter(data['arrested_fraction'], data['clustered_fraction'], 
                       alpha=0.5, s=20)
        
        ax3.plot([0, 1], [0, 1], 'k--', alpha=0.3)
        ax3.set_xlabel('Arrested Fraction')
        ax3.set_ylabel('Fraction in Clusters')
        ax3.set_title('Arrest → Clustering')
        ax3.set_xlim(0, 1)
        ax3.set_ylim(0, 1)
        ax3.grid(True, alpha=0.3)
        
        # 4. Comparison with other parameters
        ax4 = plt.subplot(2, 3, 5)
        comparison_params = [(2.0, 2.0), (2.0, 4.0), (2.0, 8.0), (3.0, 4.0)]
        
        for t, s in comparison_params:
            runs = df[(df['threshold'] == t) & (df['steepness'] == s)]
            if not runs.empty:
                mean_data = runs.iloc[0]['temporal_data']  # Use first seed
                label = f'T={t}, k={s}' + (' *' if (t, s) == optimal_params else '')
                ax4.plot(mean_data['time'], mean_data['clustered_fraction'], 
                        linewidth=3 if (t, s) == optimal_params else 1.5,
                        label=label)
        
        ax4.set_xlabel('Time')
        ax4.set_ylabel('Fraction in Arrested Clusters')
        ax4.set_title('Parameter Comparison')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        ax4.set_ylim(0, 1)
        
        # 5. Spatial snapshot at peak clustering
        ax5 = plt.subplot(2, 3, 6)
        best_run = optimal_runs.iloc[0]
        peak_idx = np.argmax(best_run['temporal_data']['clustered_fraction'])
        time_idx = peak_idx * 10  # Account for sampling
        
        positions = best_run['temporal_data']['positions'] if 'positions' in best_run['temporal_data'] else None
        states = best_run['temporal_data']['states'] if 'states' in best_run['temporal_data'] else None
        
        # For visualization, we'll create a schematic
        ax5.text(0.5, 0.5, f'Peak Clustering\nT={best_run["temporal_data"]["time"][peak_idx]:.0f}\n' + 
                f'{best_run["temporal_data"]["clustered_fraction"][peak_idx]:.1%} clustered',
                ha='center', va='center', fontsize=12)
        ax5.set_xlim(0, 1)
        ax5.set_ylim(0, 1)
        ax5.axis('off')
        
        plt.suptitle('Arrested-Only Clustering Analysis', fontsize=16)
        plt.tight_layout()
        return fig

def main():
    # Target parameters to analyse
    target_params = [
        (2.0, 2.0), (2.0, 4.0), (2.0, 8.0), (2.0, 16.0),
        (3.0, 2.0), (3.0, 4.0), (3.0, 8.0),
        (1.0, 4.0), (4.0, 4.0)
    ]
    
    # Load and analyse
    latest_dir = ArrestedClusteringAnalyser.find_latest_run(
        "/data/michael/ants/multirun/stigmergy/phase_diagram/"
    )
    analyser = ArrestedClusteringAnalyser(latest_dir)
    
    print("Loading runs...")
    runs = analyser.load_runs(target_params)
    print(f"Loaded {len(runs)} runs")
    
    print("\nAnalysing arrested clustering dynamics...")
    df = analyser.analyse_runs(runs)
    
    # Find optimal parameters
    param_summary = analyser.find_optimal_params(df)
    optimal = param_summary.index[0]  # Best scoring
    
    print("\n" + "="*60)
    print("OPTIMAL PARAMETERS FOR ARRESTED CLUSTERING")
    print("="*60)
    print(f"Threshold: {optimal[0]}, Steepness: {optimal[1]}")
    print("\nJustification:")
    print(f"- Max clustered fraction: {param_summary.iloc[0]['max_clustered']['mean']:.1%} ± {param_summary.iloc[0]['max_clustered']['std']:.1%}")
    print(f"- Final clustered fraction: {param_summary.iloc[0]['final_clustered']['mean']:.1%} ± {param_summary.iloc[0]['final_clustered']['std']:.1%}")
    score_value = param_summary['score'].iloc[0]
    print(f"- Consistency score: {score_value:.3f}")
    print("\nTop 3 parameter combinations:")
    print(param_summary.head(5))
    print("="*60)
    
    # Create figure
    fig = analyser.create_figure(df, optimal)
    fig.savefig('arrested_clustering_analysis.pdf', dpi=300, bbox_inches='tight')
    print("\nFigure saved as 'arrested_clustering_analysis.pdf'")

if __name__ == "__main__":
    main()