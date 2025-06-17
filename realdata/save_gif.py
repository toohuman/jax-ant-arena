import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import imageio

def main():
    # Load data
    data = pd.read_pickle("../../ant-info/maruyama_data/KC048_10cm_5h_20230610_1h-2h.pkl")
    num_ants = data.shape[1] // 2
    num_frames = data.shape[0]
    
    print(f"Data shape: {data.shape}")
    print(f"Number of ants: {num_ants}")
    print(f"Number of frames: {num_frames}")
    
    # Check coordinate ranges
    x_coords = data.iloc[:, 0::2].values.flatten()
    y_coords = data.iloc[:, 1::2].values.flatten()
    x_min, x_max = np.nanmin(x_coords), np.nanmax(x_coords)
    y_min, y_max = np.nanmin(y_coords), np.nanmax(y_coords)
    print(f"X coordinate range: {x_min:.2f} to {x_max:.2f}")
    print(f"Y coordinate range: {y_min:.2f} to {y_max:.2f}")
    
    # Calculate transformation parameters
    x_center = (x_min + x_max) / 2
    y_center = (y_min + y_max) / 2
    x_span = x_max - x_min
    y_span = y_max - y_min
    max_span = max(x_span, y_span)
    
    print(f"Data center: ({x_center:.2f}, {y_center:.2f})")
    print(f"Data span: X={x_span:.2f}, Y={y_span:.2f}")

    # Visualization parameters
    arena_radius = 400.0  # Leave some margin
    ant_length = 15.0  # Reasonable size for visibility
    ant_width = ant_length / 2.0
    fig_size = 8
    frames = []

    # Prepare figure
    fig, ax = plt.subplots(figsize=(fig_size, fig_size))
    ax.set_xlim(-arena_radius, arena_radius)
    ax.set_ylim(-arena_radius, arena_radius)
    ax.set_aspect('equal')
    ax.axis('off')

    # Draw arena background
    arena_circle = plt.Circle((0, 0), arena_radius, color=(230/255, 220/255, 240/255), zorder=0)
    ax.add_patch(arena_circle)

    # For each frame, draw ants and save as image (every 30th frame for speed)
    frame_step = 300
    for t in range(0, num_frames, frame_step):
        # Remove previous ant patches, keep only arena_circle
        for patch in list(ax.patches):
            if patch is not arena_circle:
                patch.remove()
        frame = data.iloc[t]
        ants_drawn = 0
        for i in range(num_ants):
            x_raw = float(frame.iloc[i * 2])
            y_raw = float(frame.iloc[i * 2 + 1])
            
            # Skip if coordinates are NaN
            if np.isnan(x_raw) or np.isnan(y_raw):
                continue
                
            # Transform coordinates to arena space
            x = ((x_raw - x_center) / max_span) * (2 * arena_radius)
            y = ((y_raw - y_center) / max_span) * (2 * arena_radius)
            
            # Compute angle if possible
            if t > 0:
                prev_x_raw = data.iloc[t - 1, i * 2]
                prev_y_raw = data.iloc[t - 1, i * 2 + 1]
                if not (np.isnan(prev_x_raw) or np.isnan(prev_y_raw)):
                    dx = x_raw - prev_x_raw
                    dy = y_raw - prev_y_raw
                    angle = np.arctan2(dy, dx)
                else:
                    angle = 0.0
            else:
                angle = 0.0

            # Ant polygon (triangle)
            ant_poly = np.array([
                [ant_length / 2, 0],
                [-ant_length / 2, -ant_width / 2],
                [-ant_length / 2, ant_width / 2]
            ])
            # Rotate and translate
            rot = np.array([
                [np.cos(angle), -np.sin(angle)],
                [np.sin(angle), np.cos(angle)]
            ])
            ant_poly = ant_poly.dot(rot) + np.array([x, y])
            ant_patch = plt.Polygon(ant_poly, closed=True, color=(1, 0, 0), zorder=1)  # Red for visibility
            ax.add_patch(ant_patch)
            ants_drawn += 1
            
        if (t // frame_step) % 100 == 0:
            print(f"Frame {t}: Drew {ants_drawn} ants")

        # Render to image
        fig.canvas.draw()
        image = np.frombuffer(fig.canvas.buffer_rgba(), dtype='uint8')
        image = image.reshape(fig.canvas.get_width_height()[::-1] + (4,))
        image = image[:, :, :3]  # Remove alpha channel
        frames.append(image)

        # Optionally, print progress
        if (t // frame_step) % 100 == 0:
            print(f"Processed frame {t}/{num_frames} (sampled frame {t // frame_step})")

    # Save as GIF
    try:
        print("Saving GIF as ant_realdata.gif ...")
        imageio.mimsave("ant_realdata.gif", frames, fps=30)
        print("Saved GIF as ant_realdata.gif")
    except Exception as e:
        print("Error saving GIF:", e)

if __name__ == "__main__":
    main()
