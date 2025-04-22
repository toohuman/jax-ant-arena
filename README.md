# 🐜〰️〰️JAX〰️Ant〰️Arena〰️〰️

JAX Ant Arena is a research tool for simulating simple pheromone-based ant colony dynamics. The simulation aims to capture the collective behavior of real (clonal) ant colonies, which exhibit strong clustering and state-switching dynamics modulated by an arrestant pheromone.

Pheromones are modeled as a global, continuously increasing function influenced by the proximity of ants to one another. Notably, pheromones are not persistent in the environment; instead, their effect is immediate and global. The simulation leverages [JAX](https://github.com/google/jax) for efficient, vectorised multi-agent modeling.

## Features
- Fast, vectorised simulation of ant colony dynamics using JAX
- Adjustable simulation speed for rapid exploration
- Real-time visualisation with matplotlib
- Easily modifiable pheromone and behavioral parameters

## Quickstart

### 1. Install [Pixi](https://prefix.dev/docs/pixi/installation) (recommended)
```sh
curl -fsSL https://pixi.sh/install.sh | bash
```

### 2. Set up the project environment
```sh
pixi install
```

Alternatively, you may use `conda` or `poetry` by installing the dependencies listed in `pixi.toml`.

### 3. Run the simulation and visualisation
```sh
pixi run visualise
```

## Project Structure
- `simulation/visualise_sim.py` — Main simulation and visualisation script
- `simulation/ant_simulation.py` — Core ant simulation logic and parameters
- `pixi.toml` — Project environment and dependency configuration

## Customisation

Simulation and pheromone parameters can be adjusted to explore different behaviors. Key parameters in `simulation/ant_simulation.py` include:
- `PHEROMONE_MAX_TIMESTEP`: Maximum time for pheromone effect
- `PHEROMONE_ELU_TRANSITION_FRAC`: Fraction of max time where pheromone growth becomes linear
- `PHEROMONE_ELU_STEEPNESS`: Controls the initial exponential rise of pheromone effect
- `MAX_PHEROMONE_STRENGTH`: Maximum contribution of a single ant

Feel free to experiment with these and other parameters to investigate different collective behaviors.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Citation

If you use this tool in your research, please cite this repository. (Add citation details here if available.)

---

For questions, contributions, or issues, please open an issue or pull request on GitHub.
