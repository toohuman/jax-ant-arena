import os
import sys
import time # For performance timing
import jax
import jax.numpy as jnp
import jax.random as random
import hydra
from omegaconf import DictConfig, OmegaConf
from functools import partial
import logging

# --- Import from ant_simulation.py ---
try:
    import ant_simulation as antsim
except ImportError:
    # Fallback if running directly and path issues occur, though module execution is preferred
    from simulation import ant_simulation as antsim


# Configure logging
logger = logging.getLogger(__name__) # Using __name__ is standard

def run_simulation_headless(cfg: DictConfig):
    """
    Runs the ant simulation without visualisation and collects data.
    """
    logger.info("Starting headless simulation run...")
    # Convert OmegaConf to a regular dictionary for easier processing
    # resolve=True handles any interpolations in the config
    params = OmegaConf.to_container(cfg, resolve=True)
    logger.info(f"Effective parameters for simulation:\n{params}")


    # --- Prepare Derived Parameters ---
    params['ant_width'] = params['ant_length'] / 2.0
    # 'ant_radius' often refers to half of ant_length for collision/interaction distance
    params['ant_radius'] = params['ant_length'] / 2.0 
    params['wall_zone_width'] = params['ant_length'] * params['wall_zone_width_multiplier']

    # --- Pheromone-specific derived parameters ---
    # Ensure 'pheromones' sub-dictionary exists
    if 'pheromones' not in params:
        logger.error("Pheromones configuration missing!")
        raise ValueError("Pheromones configuration ('params.pheromones') is missing.")

    if params['pheromones'].get('use_grid_pheromones', False): # Default to False if key is missing
        logger.info("Grid pheromone mode selected.")
        if params['pheromones'].get('grid_resolution', 0) > 0:
            params['grid_cell_size'] = 2.0 * params['arena_radius'] / params['pheromones']['grid_resolution']
            
            # Prioritise directly set grid_radius_cells if available and valid
            current_grid_radius_cells = params['pheromones'].get('grid_radius_cells', 0)
            if not (isinstance(current_grid_radius_cells, int) and current_grid_radius_cells > 0):
                logger.warning("'grid_radius_cells' not directly set or invalid in config for grid mode. Attempting to derive it.")
                world_pheromone_radius = 0.0
                if 'pheromone_radius_multiplier' in params['pheromones']:
                    world_pheromone_radius = params['ant_length'] * params['pheromones']['pheromone_radius_multiplier']
                    params['pheromones']['pheromone_radius'] = world_pheromone_radius # Store for info
                elif 'pheromone_radius' in params['pheromones']:
                    world_pheromone_radius = params['pheromones']['pheromone_radius']
                else:
                    logger.error("Cannot determine world_pheromone_radius to derive grid_radius_cells. Please define 'grid_radius_cells', 'pheromone_radius', or 'pheromone_radius_multiplier' in grid config.")
                    # Assign a default/fallback if absolutely necessary, or raise error
                    params['pheromones']['grid_radius_cells'] = 1 # Small fallback
                
                if world_pheromone_radius > 0 and params['grid_cell_size'] > 0:
                    params['pheromones']['grid_radius_cells'] = int(jnp.ceil(world_pheromone_radius / params['grid_cell_size']))
                elif params['pheromones'].get('grid_radius_cells', 0) <= 0 : # If still not set
                     params['pheromones']['grid_radius_cells'] = 1 # Default small value
                     logger.error(f"grid_cell_size ({params['grid_cell_size']}) or world_pheromone_radius ({world_pheromone_radius}) invalid for grid_radius_cells derivation.")
            
            # Ensure pheromone_radius (world units) is consistent if grid_radius_cells was set
            if 'pheromone_radius' not in params['pheromones'] and \
               params['pheromones'].get('grid_radius_cells', 0) > 0 and \
               params.get('grid_cell_size', 0) > 0:
                 params['pheromones']['pheromone_radius'] = params['pheromones']['grid_radius_cells'] * params['grid_cell_size']
                 logger.info(f"Derived world pheromone_radius for grid mode: {params['pheromones']['pheromone_radius']}")

        else: # grid_resolution is 0, but use_grid_pheromones is true (inconsistent config)
            logger.error("'use_grid_pheromones' is true, but 'grid_resolution' is 0 or undefined. Disabling grid pheromones.")
            params['pheromones']['use_grid_pheromones'] = False 
            params['grid_cell_size'] = 0.0
            params['pheromones']['grid_radius_cells'] = 0
            # Fall through to direct mode logic for pheromone_radius setup
            
    if not params['pheromones'].get('use_grid_pheromones', False): # Direct mode or fallback
        logger.info("Direct pheromone mode selected (or fallback from inconsistent grid config).")
        if 'pheromone_radius' not in params['pheromones']:
            if 'pheromone_radius_multiplier' in params['pheromones']:
                params['pheromones']['pheromone_radius'] = params['ant_length'] * params['pheromones']['pheromone_radius_multiplier']
            else:
                default_direct_radius = params['ant_length'] * 2.0 # Default fallback
                logger.error(f"Cannot determine 'pheromone_radius' for direct mode. Please define 'pheromone_radius' or 'pheromone_radius_multiplier'. Using fallback: {default_direct_radius}")
                params['pheromones']['pheromone_radius'] = default_direct_radius
        
        # These are not functionally used by direct mode simulation logic
        params['grid_cell_size'] = 0.0 
        params['pheromones']['grid_radius_cells'] = 0
        if 'grid_resolution' not in params['pheromones']: # Ensure it exists for ant_simulation if accessed early
            params['pheromones']['grid_resolution'] = 0


    # --- JAX Simulation Setup ---
    key = random.PRNGKey(cfg.experiment.seed) 
    key, subkey = random.split(key)
    
    sim_state = antsim.initialise_state(subkey, params)
    
    update_fn = partial(antsim.update_step, params=params)
    jitted_update_fn = jax.jit(update_fn)

    sim_time = 0.0
    dt = params['dt']
    max_sim_time = cfg.experiment.max_simulation_time
    data_collection_interval_time = cfg.experiment.data_collection_interval
    
    steps_per_collection = max(1, int(round(data_collection_interval_time / dt))) # round for better precision
    data_collection_interval_actual_time = steps_per_collection * dt
    
    logger.info(f"Simulation time step (dt): {dt}")
    logger.info(f"Max simulation time: {max_sim_time}")
    logger.info(f"Data collection interval (steps): {steps_per_collection} (actual time: {data_collection_interval_actual_time:.2f}s)")

    # --- Data Collection Setup ---
    collected_data_lists = {
        'time': [],
        'positions': [],      
        'angles': [],         
        'behavioural_states': [],
    }
    if params['pheromones'].get('use_grid_pheromones', False) and cfg.experiment.collect_pheromone_map:
        collected_data_lists['pheromone_maps'] = []

    num_steps = int(round(max_sim_time / dt))
    next_collection_step = 0
    
    start_real_time = time.perf_counter()

    for step_idx in range(num_steps + 1):
        current_sim_time_for_step = step_idx * dt

        # --- Data Collection ---
        if step_idx >= next_collection_step:
            logger.info(f"Collecting data at step {step_idx}, sim_time: {current_sim_time_for_step:.2f}s")
            collected_data_lists['time'].append(current_sim_time_for_step)
            
            positions_host = jax.device_get(sim_state['position'])
            angles_host = jax.device_get(sim_state['angle'])
            behavioural_states_host = jax.device_get(sim_state['behavioural_state'])
            
            collected_data_lists['positions'].append(positions_host)
            collected_data_lists['angles'].append(angles_host)
            collected_data_lists['behavioural_states'].append(behavioural_states_host)
            
            if params['pheromones'].get('use_grid_pheromones', False) and cfg.experiment.collect_pheromone_map:
                pheromone_map_host = jax.device_get(sim_state['pheromone_map'])
                collected_data_lists['pheromone_maps'].append(pheromone_map_host)
            
            next_collection_step += steps_per_collection

        if step_idx == num_steps: 
            break

        # --- Simulation Step ---
        key, subkey = random.split(key)
        sim_state = jitted_update_fn(sim_state, subkey, current_sim_time_for_step) # Pass current sim time for this step
        
        if step_idx > 0 and step_idx % max(1, (num_steps // 20)) == 0 : # Log roughly 20 times
             current_real_time = time.perf_counter()
             elapsed_real_time = current_real_time - start_real_time
             time_per_step = elapsed_real_time / step_idx
             remaining_steps = num_steps - step_idx
             eta_seconds = time_per_step * remaining_steps
             logger.info(f"Progress: {step_idx / num_steps * 100:.1f}%, Sim Time: {current_sim_time_for_step:.2f}s, ETA: {eta_seconds:.1f}s")

    end_real_time = time.perf_counter()
    total_real_time = end_real_time - start_real_time
    final_sim_time = (num_steps) * dt
    logger.info(f"Simulation finished. Total steps: {num_steps}, Final Sim time: {final_sim_time:.2f}s")
    logger.info(f"Total real time for simulation: {total_real_time:.2f}s")

    # --- Save Collected Data ---
    output_filename_base = "simulation_data"
    # Hydra sets the working directory to the output folder for each run
    save_path_base = os.getcwd() 
    
    data_to_save_numpy = {}
    for key_data, val_list in collected_data_lists.items():
        if val_list: # Ensure list is not empty
            if key_data == 'time': # Time is just a list of scalars
                 data_to_save_numpy[key_data] = jnp.array(val_list)
            else: # Other data are lists of arrays, stack them
                 data_to_save_numpy[key_data] = jnp.stack(val_list, axis=0)
        else:
            data_to_save_numpy[key_data] = jnp.array([]) # Save empty array if no data collected

    output_file = os.path.join(save_path_base, f"{output_filename_base}.npz")
    jnp.savez(output_file, **data_to_save_numpy)
    logger.info(f"Collected data saved to: {output_file}")


@hydra.main(config_path="../conf", config_name="config", version_base=None)
def main(cfg: DictConfig):
    # Actual run is handled by the decorated function if no sweep
    # For sweeps, Hydra calls this multiple times with varied cfg
    run_simulation_headless(cfg)

if __name__ == "__main__":
    # This allows running the script directly, e.g., for debugging
    # `python simulation/py` from project root
    # Hydra configurations can be overridden from command line
    # e.g. `python simulation/py num_ants=10 seed=42 max_simulation_time=100`
    main()