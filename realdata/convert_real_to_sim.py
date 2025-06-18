import numpy as np
import pandas as pd
import os
import pickle
from pathlib import Path

def calculate_moving_average_direction(x_coords, y_coords, window_size=5):
    """Calculate direction using moving average of x,y coordinates"""
    directions = np.zeros(len(x_coords))
    
    for i in range(len(x_coords)):
        if i < window_size or i >= len(x_coords) - window_size:
            # For edge cases, use simple difference
            if i > 0 and i < len(x_coords) - 1:
                dx = x_coords[i + 1] - x_coords[i - 1]
                dy = y_coords[i + 1] - y_coords[i - 1]
            elif i == 0 and len(x_coords) > 1:
                dx = x_coords[i + 1] - x_coords[i]
                dy = y_coords[i + 1] - y_coords[i]
            elif i == len(x_coords) - 1 and len(x_coords) > 1:
                dx = x_coords[i] - x_coords[i - 1]
                dy = y_coords[i] - y_coords[i - 1]
            else:
                dx, dy = 0, 0
        else:
            # Use moving average for direction calculation
            start_idx = i - window_size
            end_idx = i + window_size + 1
            
            x_smooth = np.mean(x_coords[start_idx:end_idx])
            y_smooth = np.mean(y_coords[start_idx:end_idx])
            
            # Calculate direction based on smoothed trajectory
            if i > window_size:
                prev_start = i - window_size - window_size
                prev_end = i - window_size + 1
                prev_x_smooth = np.mean(x_coords[prev_start:prev_end])
                prev_y_smooth = np.mean(y_coords[prev_start:prev_end])
                
                dx = x_smooth - prev_x_smooth
                dy = y_smooth - prev_y_smooth
            else:
                dx, dy = 0, 0
        
        # Calculate angle
        directions[i] = np.arctan2(dy, dx)
    
    return directions

def convert_real_to_sim_data(data_file_path, output_dir="real2sim_data"):
    """Convert real ant data to simulation format"""
    
    print(f"Loading data from {data_file_path}...")
    data = pd.read_pickle(data_file_path)
    
    num_ants = data.shape[1] // 2
    num_frames = data.shape[0]
    
    print(f"Data shape: {data.shape}")
    print(f"Number of ants: {num_ants}")
    print(f"Number of frames: {num_frames}")
    
    # Get coordinate ranges
    x_coords = data.iloc[:, 0::2].values.flatten()
    y_coords = data.iloc[:, 1::2].values.flatten()
    
    # Remove NaN values for range calculation
    valid_x = x_coords[~np.isnan(x_coords)]
    valid_y = y_coords[~np.isnan(y_coords)]
    
    x_min, x_max = np.min(valid_x), np.max(valid_x)
    y_min, y_max = np.min(valid_y), np.max(valid_y)
    
    print(f"X coordinate range: {x_min:.2f} to {x_max:.2f}")
    print(f"Y coordinate range: {y_min:.2f} to {y_max:.2f}")
    
    # Calculate transformation parameters for arena (assuming circular arena with radius 500)
    x_center = (x_min + x_max) / 2
    y_center = (y_min + y_max) / 2
    x_span = x_max - x_min
    y_span = y_max - y_min
    max_span = max(x_span, y_span)
    
    arena_radius = 400.0  # Leave some margin
    
    print(f"Data center: ({x_center:.2f}, {y_center:.2f})")
    print(f"Max span: {max_span:.2f}")
    
    # Create output directory
    Path(output_dir).mkdir(exist_ok=True)
    
    # Convert data frame by frame
    converted_positions = np.zeros((num_frames, num_ants, 2))  # x, y for each ant
    converted_directions = np.zeros((num_frames, num_ants))    # direction for each ant
    
    print("Converting coordinates and calculating directions...")
    
    for ant_id in range(num_ants):
        print(f"Processing ant {ant_id + 1}/{num_ants}...")
        
        # Extract coordinates for this ant
        x_col = ant_id * 2
        y_col = ant_id * 2 + 1
        
        x_raw = data.iloc[:, x_col].values
        y_raw = data.iloc[:, y_col].values
        
        # Transform coordinates to arena space
        x_transformed = ((x_raw - x_center) / max_span) * (2 * arena_radius)
        y_transformed = ((y_raw - y_center) / max_span) * (2 * arena_radius)
        
        # Calculate directions using moving average
        valid_indices = ~(np.isnan(x_raw) | np.isnan(y_raw))
        
        if np.any(valid_indices):
            # Only calculate directions for valid coordinates
            valid_x = x_raw[valid_indices]
            valid_y = y_raw[valid_indices]
            
            if len(valid_x) > 1:
                valid_directions = calculate_moving_average_direction(valid_x, valid_y)
                
                # Map back to full array
                directions_full = np.zeros(num_frames)
                directions_full[valid_indices] = valid_directions
                
                # For invalid coordinates, use NaN
                directions_full[~valid_indices] = np.nan
            else:
                directions_full = np.full(num_frames, np.nan)
        else:
            directions_full = np.full(num_frames, np.nan)
        
        # Store converted data
        converted_positions[:, ant_id, 0] = x_transformed
        converted_positions[:, ant_id, 1] = y_transformed
        converted_directions[:, ant_id] = directions_full
    
    # Save metadata
    metadata = {
        'num_ants': num_ants,
        'num_frames': num_frames,
        'arena_radius': arena_radius,
        'original_data_range': {
            'x_min': x_min, 'x_max': x_max,
            'y_min': y_min, 'y_max': y_max
        },
        'transformation_params': {
            'x_center': x_center,
            'y_center': y_center,
            'max_span': max_span
        }
    }
    
    # Save all data
    print("Saving converted data...")
    
    with open(f"{output_dir}/positions.pkl", 'wb') as f:
        pickle.dump(converted_positions, f)
    
    with open(f"{output_dir}/directions.pkl", 'wb') as f:
        pickle.dump(converted_directions, f)
    
    with open(f"{output_dir}/metadata.pkl", 'wb') as f:
        pickle.dump(metadata, f)
    
    print(f"Conversion complete! Data saved to {output_dir}/")
    print(f"- positions.pkl: {converted_positions.shape}")
    print(f"- directions.pkl: {converted_directions.shape}")
    print(f"- metadata.pkl: metadata dictionary")
    
    return converted_positions, converted_directions, metadata

if __name__ == "__main__":
    # Path to the real data
    data_file = "../../ant-info/maruyama_data/KC048_10cm_5h_20230610_1h-2h.pkl"
    
    # Convert the data
    positions, directions, metadata = convert_real_to_sim_data(data_file)
