import numpy as np
import pickle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

def load_converted_data(data_dir="real2sim_data"):
    """Load converted simulation data"""
    
    print(f"Loading converted data from {data_dir}...")
    
    with open(f"{data_dir}/positions.pkl", 'rb') as f:
        positions = pickle.load(f)
    
    with open(f"{data_dir}/directions.pkl", 'rb') as f:
        directions = pickle.load(f)
    
    with open(f"{data_dir}/metadata.pkl", 'rb') as f:
        metadata = pickle.load(f)
    
    print(f"Loaded data:")
    print(f"- Positions shape: {positions.shape}")
    print(f"- Directions shape: {directions.shape}")
    print(f"- Number of ants: {metadata['num_ants']}")
    print(f"- Number of frames: {metadata['num_frames']}")
    print(f"- Arena radius: {metadata['arena_radius']}")
    
    return positions, directions, metadata

def create_snapshot(positions, directions, frame_idx, metadata, title=""):
    """Create a single snapshot visualization"""
    
    arena_radius = metadata['arena_radius']
    num_ants = metadata['num_ants']
    
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_xlim(-arena_radius, arena_radius)
    ax.set_ylim(-arena_radius, arena_radius)
    ax.set_aspect('equal')
    ax.set_title(f"{title} (Frame {frame_idx})", fontsize=16)
    
    # Draw arena background
    arena_circle = plt.Circle((0, 0), arena_radius, 
                             color=(0.9, 0.9, 0.95), 
                             fill=True, zorder=0)
    ax.add_patch(arena_circle)
    
    # Draw arena border
    arena_border = plt.Circle((0, 0), arena_radius, 
                             color='black', fill=False, 
                             linewidth=2, zorder=1)
    ax.add_patch(arena_border)
    
    # Count valid ants for this frame
    valid_ants = 0
    
    # Draw ants
    for ant_id in range(num_ants):
        x = positions[frame_idx, ant_id, 0]
        y = positions[frame_idx, ant_id, 1]
        direction = directions[frame_idx, ant_id]
        
        # Skip if coordinates are NaN
        if np.isnan(x) or np.isnan(y) or np.isnan(direction):
            continue
        
        valid_ants += 1
        
        # Ant body (circle)
        ant_body = plt.Circle((x, y), 8, color='red', zorder=2)
        ax.add_patch(ant_body)
        
        # Ant direction indicator (arrow)
        arrow_length = 15
        dx = np.cos(direction) * arrow_length
        dy = np.sin(direction) * arrow_length
        
        ax.arrow(x, y, dx, dy, 
                head_width=5, head_length=3, 
                fc='darkred', ec='darkred', zorder=3)
    
    # Add info text
    ax.text(-arena_radius + 20, arena_radius - 30, 
            f"Valid ants: {valid_ants}/{num_ants}", 
            fontsize=12, bbox=dict(boxstyle="round,pad=0.3", 
                                  facecolor="white", alpha=0.8))
    
    # Remove axis ticks and labels
    ax.set_xticks([])
    ax.set_yticks([])
    ax.axis('off')
    
    return fig

def visualise_three_snapshots(data_dir="real2sim_data", output_dir="visualize"):
    """Create three snapshots from the converted data"""
    
    # Load data
    positions, directions, metadata = load_converted_data(data_dir)
    
    num_frames = metadata['num_frames']
    
    # Select three time points: start, middle, end
    frame_indices = [
        0,                          # Start
        num_frames // 2,           # Middle  
        num_frames - 1             # End
    ]
    
    frame_names = ["Start", "Middle", "End"]
    
    print(f"Creating snapshots at frames: {frame_indices}")
    
    # Create output directory
    Path(output_dir).mkdir(exist_ok=True)
    
    # Generate and save snapshots
    for i, (frame_idx, frame_name) in enumerate(zip(frame_indices, frame_names)):
        print(f"Creating {frame_name} snapshot (frame {frame_idx})...")
        
        fig = create_snapshot(positions, directions, frame_idx, metadata, 
                            title=frame_name)
        
        # Save the figure
        output_file = f"{output_dir}/snapshot_{i+1:02d}_{frame_name.lower()}.png"
        fig.savefig(output_file, dpi=150, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        plt.close(fig)
        
        print(f"Saved: {output_file}")
    
    print(f"\nVisualization complete! Snapshots saved to {output_dir}/")
    
    # Print summary statistics
    print("\n=== Data Summary ===")
    
    # Calculate statistics for each snapshot
    for i, (frame_idx, frame_name) in enumerate(zip(frame_indices, frame_names)):
        frame_positions = positions[frame_idx]
        frame_directions = directions[frame_idx]
        
        # Count valid ants
        valid_mask = ~(np.isnan(frame_positions[:, 0]) | 
                      np.isnan(frame_positions[:, 1]) | 
                      np.isnan(frame_directions))
        
        valid_count = np.sum(valid_mask)
        
        if valid_count > 0:
            # Calculate position statistics
            valid_positions = frame_positions[valid_mask]
            distances = np.sqrt(valid_positions[:, 0]**2 + valid_positions[:, 1]**2)
            
            print(f"{frame_name} (Frame {frame_idx}):")
            print(f"  Valid ants: {valid_count}/{metadata['num_ants']}")
            print(f"  Average distance from center: {np.mean(distances):.1f}")
            print(f"  Max distance from center: {np.max(distances):.1f}")
        else:
            print(f"{frame_name} (Frame {frame_idx}): No valid ants")

if __name__ == "__main__":
    # Create visualization
    visualise_three_snapshots()
