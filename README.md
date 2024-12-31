# GridWorld Visualisation (Sutton & Barto Example 3.5)

This project provides a Python-based visualization of the **GridWorld environment** described in Example 3.5 from Sutton and Barto's *Reinforcement Learning: An Introduction*. The project demonstrates how state values and policies converge during Q-learning and generates visual representations of these concepts.

## Features

- **GridWorld Environment**: A 5x5 grid with special transitions and rewards:
  - **State A** transitions to A' with a reward of +10.
  - **State B** transitions to B' with a reward of +5.
- **Q-Learning Training**: Trains an agent to navigate the environment and optimize its policy.
- **Visualizations**:
  - Heatmaps of state values.
  - Arrow plots of the corresponding policy.
  - An optional GIF showcasing the convergence of values and policy over episodes.

## Directory Structure

```
gridworld-visualisation/
├── agent.py            # Implementation of the Q-learning agent
├── gridworld_env.py    # Definition of the GridWorld environment
├── main.py             # Entry point for training and visualization
├── train.py            # Handles Q-learning training and optional GIF creation
├── visualisation.py    # Visualization utilities for values and policy
```

## Installation

### Requirements
- Python 3.7+
- Required Python libraries:
  - `numpy`
  - `matplotlib`
  - `Pillow`

Install the required libraries using pip:

```bash
pip install numpy matplotlib pillow
```

## Usage

1. Clone this repository:

```bash
git clone https://github.com/yourusername/gridworld-visualisation.git
cd gridworld-visualisation
```

2. Run the main script:

```
python main.py
```

### Outputs

- **Convergence GIF**: A file named `convergence.gif` visualizing how the state values and policy evolve during training.
- **Final Policy Image**: A file named `final_values_and_policy.png` showing the final state values and optimal policy.

### Example Visuals

- **Final Values and Policy**:
  - Left: Heatmap of the final state values.
  - Right: Arrows representing the optimal policy.

> **Note**: Visuals are based on Example 3.5 from *Reinforcement Learning: An Introduction* by Sutton and Barto.

## How It Works

### Training

The Q-learning agent is trained on the GridWorld environment over a number of episodes. During training, the Q-values are updated using the Q-learning update rule, and the state values V(s) = max_a Q(s, a) are periodically recorded.

### Visualization

The `visualisation.py` module generates:
- Heatmaps for state values.
- Arrow plots for policies based on the Q-table.
- A GIF showing the convergence of state values and policy over episodes (optional).

## Customization

- **Number of Episodes**: Modify `num_episodes` in `main.py` or `train.py` to adjust the training duration.
- **Output Files**: Change `gif_filename` or the save paths in `main.py` to customize output filenames.
- **Environment and Agent Parameters**: Adjust parameters like the grid size, discount factor, or exploration rate by editing `gridworld_env.py` and `agent.py`.

## References

This implementation is based on Example 3.5 from *Reinforcement Learning: An Introduction* by Richard S. Sutton and Andrew G. Barto. For more details, refer to [their official book](http://incompleteideas.net/book/the-book-2nd.html).
