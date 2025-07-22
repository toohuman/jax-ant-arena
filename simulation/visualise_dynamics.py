
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from omegaconf import OmegaConf
from sklearn.cluster import DBSCAN
import seaborn as sns
import itertools

# Set publication-quality defaults
plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10

STATE_MAP = {0: 'WANDERING', 1: 'DEPOSITING', 2: 'ARRESTED'}
STATE_COLORS = {'WANDERING': 'blue', 'DEPOSITING': 'orange', 'ARRESTED': 'green'}

class DynamicsVisualiser:
    def __init__(self, multirun_dir, output_dir):
        self.multirun_dir = Path(multirun_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        if not self.multirun_dir.is_dir():
            raise ValueError(f"Directory not found: {self.multirun_dir}")
        self.data = None
        self.analysis_results = None
        self.arena_radius = 50.0
        self.varied_params = []

    def load_data(self):
        all_run_data = []
        run_dirs = [d for d in self.multirun_dir.iterdir() if d.is_dir() and d.name.isdigit()]
        print(f"Found {len(run_dirs)} run directories in {self.multirun_dir}")

        # First pass to find varied parameters
        param_values = {}
        all_params = []

        for run_dir in run_dirs:
            params = {}
            metadata_file = run_dir / ".hydra/overrides.yaml"
            if metadata_file.exists():
                cfg = OmegaConf.load(metadata_file)
                params = {item.split('=')[0].lstrip('+'): item.split('=')[1] for item in cfg}
            else:
                # Fallback for older structure
                metadata_file = run_dir / "metadata.yaml"
                if metadata_file.exists():
                    cfg = OmegaConf.load(metadata_file)
                    # Ensure params are flattened from metadata.yaml as well
                    raw_params = cfg.get('params', {})
                    if raw_params:
                        params = pd.json_normalize(raw_params, sep='.').to_dict(orient='records')[0]

            all_params.append(params)
            for key, value in params.items():
                if key not in param_values:
                    param_values[key] = set()
                param_values[key].add(value)

        self.varied_params = [k for k, v in param_values.items() if len(v) > 1 and k != 'seed']
        if not self.varied_params:
            self.varied_params = [k for k in param_values.keys() if k != 'seed']

        print(f"Identified varied parameters: {self.varied_params}")

        for i, run_dir in enumerate(run_dirs):
            data_file = run_dir / "simulation_data.npz"
            if not data_file.exists():
                continue

            sim_data = np.load(data_file)
            
            run_df = pd.DataFrame({
                'time': sim_data['time'],
                'positions': list(sim_data['positions']),
                'behavioural_states': list(sim_data['behavioural_states'])
            })
            
            # Add parameters from our collected list
            params = all_params[i]
            for key, value in params.items():
                # Try to convert to numeric if possible
                try:
                    run_df[key] = pd.to_numeric(value)
                except (ValueError, TypeError):
                    run_df[key] = value
            
            run_df['run_id'] = run_dir.name
            all_run_data.append(run_df)

        if not all_run_data:
            print("No data loaded. Exiting.")
            return False

        self.data = pd.concat(all_run_data, ignore_index=True)
        # Fallback if a varied param is still not a column
        for param in self.varied_params:
            if param not in self.data.columns:
                print(f"Warning: Varied parameter '{param}' not found in DataFrame columns. Removing from groupby.")
                self.varied_params.remove(param)
        
        print("Data loaded successfully. DataFrame shape:", self.data.shape)
        return True

    def _analyse_timestep_states(self, states_array):
        n_ants = len(states_array)
        counts = np.bincount(states_array, minlength=len(STATE_MAP))
        return pd.Series({f'{STATE_MAP[i].lower()}_frac': count / n_ants for i, count in enumerate(counts)})

    def _analyse_timestep_clusters(self, row, eps=2.5, min_samples=3):
        arrested_mask = (row['behavioural_states'] == 2)
        arrested_positions = row['positions'][arrested_mask]
        
        if len(arrested_positions) < min_samples:
            return pd.Series({'n_clusters': 0, 'largest_cluster_size': 0, 'largest_cluster_fraction': 0.0})

        db = DBSCAN(eps=eps, min_samples=min_samples).fit(arrested_positions)
        labels = db.labels_
        
        unique_labels = set(labels)
        n_clusters = len(unique_labels) - (1 if -1 in labels else 0)
        
        if n_clusters == 0:
            return pd.Series({'n_clusters': 0, 'largest_cluster_size': 0, 'largest_cluster_fraction': 0.0})

        cluster_sizes = [np.sum(labels == label) for label in unique_labels if label != -1]
        largest_cluster_size = max(cluster_sizes) if cluster_sizes else 0
        total_arrested = len(arrested_positions)
        largest_cluster_fraction = largest_cluster_size / total_arrested if total_arrested > 0 else 0.0
        
        return pd.Series({'n_clusters': n_clusters, 'largest_cluster_size': largest_cluster_size, 'largest_cluster_fraction': largest_cluster_fraction})

    def perform_analysis(self):
        if self.data is None:
            print("No data to analyse.")
            return
        print("Analysing state proportions...")
        state_analysis = self.data['behavioural_states'].apply(self._analyse_timestep_states)
        print("Analysing cluster dynamics...")
        cluster_analysis = self.data.apply(self._analyse_timestep_clusters, axis=1)
        self.analysis_results = pd.concat([self.data, state_analysis, cluster_analysis], axis=1)
        print("Analysis complete.")

    def plot_state_proportions(self, data_group, output_path, params_title):
        fig, ax = plt.subplots(figsize=(10, 6))
        for state_name in STATE_MAP.values():
            col_name = f'{state_name.lower()}_frac'
            sns.lineplot(data=data_group, x='time', y=col_name, ax=ax, label=state_name, errorbar='sd')
        ax.set_title(f'State Proportions Over Time\n({params_title})')
        ax.set_xlabel('Time (steps)')
        ax.set_ylabel('Fraction of Ants')
        ax.legend(title='State')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1)
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()

    def plot_cluster_dynamics(self, data_group, output_path, params_title):
        fig, axes = plt.subplots(3, 1, figsize=(10, 15), sharex=True)
        fig.suptitle(f'Cluster Dynamics Over Time\n({params_title})', fontsize=16)
        
        sns.lineplot(data=data_group, x='time', y='n_clusters', ax=axes[0], errorbar='sd')
        axes[0].set_title('Number of Clusters')
        sns.lineplot(data=data_group, x='time', y='largest_cluster_size', ax=axes[1], errorbar='sd')
        axes[1].set_title('Size of Largest Cluster')
        sns.lineplot(data=data_group, x='time', y='largest_cluster_fraction', ax=axes[2], errorbar='sd')
        axes[2].set_title('Fraction of Arrested Ants in Largest Cluster')
        axes[2].set_ylim(0, 1)

        for ax in axes:
            ax.grid(True, alpha=0.3)
        axes[-1].set_xlabel('Time (steps)')
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(output_path)
        plt.close()

    def plot_spatial_snapshot(self, data_group, time_step, output_path, params_title):
        # Use the first run_id in the group for a representative snapshot
        rep_run_id = data_group['run_id'].iloc[0]
        snapshot_data = data_group[(data_group['run_id'] == rep_run_id) & (data_group['time'] == time_step)]
        
        if snapshot_data.empty:
            return
        snapshot = snapshot_data.iloc[0]
        positions, states = snapshot['positions'], snapshot['behavioural_states']
        
        fig, ax = plt.subplots(figsize=(8, 8))
        colors = [STATE_COLORS[STATE_MAP[s]] for s in states]
        ax.scatter(positions[:, 0], positions[:, 1], c=colors, alpha=0.7, s=30)
        circle = plt.Circle((0, 0), self.arena_radius, fill=False, linestyle='--', color='gray')
        ax.add_patch(circle)
        ax.set_title(f'Spatial Snapshot at Time={time_step}\n({params_title})')
        ax.set_aspect('equal')
        ax.set_xlim(-self.arena_radius - 5, self.arena_radius + 5)
        ax.set_ylim(-self.arena_radius - 5, self.arena_radius + 5)
        from matplotlib.lines import Line2D
        legend_elements = [Line2D([0], [0], marker='o', color='w', label=s, markerfacecolor=c, markersize=10) for s, c in STATE_COLORS.items()]
        ax.legend(handles=legend_elements, title="State")
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()

    def run_full_analysis(self):
        if not self.load_data():
            return
        self.perform_analysis()

        if not self.varied_params:
            print("No varied parameters found to group by. Generating overall analysis.")
            self.plot_state_proportions(self.analysis_results, self.output_dir / "state_proportions.png", "All Runs")
            self.plot_cluster_dynamics(self.analysis_results, self.output_dir / "cluster_dynamics.png", "All Runs")
            return

        grouped = self.analysis_results.groupby(self.varied_params)
        print(f"\nFound {len(grouped)} parameter combinations. Generating plots for each...")

        for params, group_df in grouped:
            params_list = params if isinstance(params, tuple) else (params,)
            param_str = "_".join(f"{k.split('.')[-1]}_{v}" for k, v in zip(self.varied_params, params_list))
            params_title = ", ".join(f"{k.split('.')[-1]}={v}" for k, v in zip(self.varied_params, params_list))
            
            group_output_dir = self.output_dir / param_str
            group_output_dir.mkdir(exist_ok=True)
            
            print(f"  Processing group: {params_title}")

            # Generate and save plots for the group
            self.plot_state_proportions(group_df, group_output_dir / "state_proportions.png", params_title)
            self.plot_cluster_dynamics(group_df, group_output_dir / "cluster_dynamics.png", params_title)
            
            max_time = group_df['time'].max()
            for t in [0, max_time // 2, max_time]:
                self.plot_spatial_snapshot(group_df, t, group_output_dir / f"snapshot_t{t}.png", params_title)
        
        print("\nGrouped analysis complete.")

def main():
    parser = argparse.ArgumentParser(description="Visualise dynamics from a Hydra multirun, grouped by parameters.")
    parser.add_argument("multirun_dir", type=str, help="Path to the Hydra multirun directory.")
    parser.add_argument("--output_dir", type=str, default="analysis_output_grouped", help="Directory to save analysis plots.")
    args = parser.parse_args()

    visualiser = DynamicsVisualiser(args.multirun_dir, args.output_dir)
    visualiser.run_full_analysis()

if __name__ == "__main__":
    main()
