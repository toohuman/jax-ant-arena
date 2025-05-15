# 🐜〰️JAX〰️Ant〰️Arena〰️〰️

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

By default, this will run the direct pheromone model (ants detect neighbours within a given radius). You may also experiment with the stigmergy model by running:
```sh
pixi run visualise pheromones=stigmergy
```

Feel free to experiment with these and other parameters to investigate different collective behaviors.

## Configuration with Hydra

This project utilises [Hydra](https://hydra.cc/) for managing configurations. The main configuration entry point is `conf/config.yaml`.

### Defaults
The `conf/config.yaml` file specifies default configurations. For instance, it includes:
```yaml
defaults:
  - hydra: interactive
  - pheromones: stigmergy
  - _self_
```
This means that by default, the `stigmergy` pheromone model is loaded from `conf/pheromones/stigmergy.yaml`.

### Pheromone Models
Pheromone model configurations are located in the `conf/pheromones/` directory. You can find files such as:
- `direct.yaml`
- `stigmergy.yaml`
- `none.yaml`

You can override the default pheromone model by specifying it on the command line when running a script that uses Hydra (like `simulation/batch_run.py` or `visualise.py`). For example:
```sh
pixi run visualise pheromones=direct
# or for batch runs
# python simulation/batch_run.py pheromones=direct
```

### Experiment Configurations
Specific experimental setups, including parameter sweeps or distinct test configurations, are defined in the `conf/experiment/` directory. Examples include:
- `direct_test.yaml`
- `stigmergy_decay_threshold_sweep.yaml`
- `direct_clustering_sweep.yaml`

These can be selected by overriding the `experiment` group in the command line. For example, to run a batch simulation with the `direct_test` configuration:
```sh
# python simulation/batch_run.py experiment=direct_test
```
If an experiment configuration also specifies a pheromone model, that will take precedence. Otherwise, the default from `config.yaml` or a command-line `pheromones=` override will be used.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Citation

If you use this tool in your research, please cite this repository. (Add citation details here if available.)

---

For questions, contributions, or issues, please open an issue or pull request on GitHub.
