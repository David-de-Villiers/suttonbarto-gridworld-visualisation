import numpy as np
from PIL import Image

from gridworld_env import GridworldEnv
from agent import QLearningAgent
from visualisation import plot_values_and_policy, fig_to_pil

def train_q_learning(num_episodes=200, make_gif=True, gif_filename="convergence.gif"):
    """
    Train the QLearningAgent on the 5x5 Gridworld. 
    After each episode, record the value function V(s).
    Optionally create a GIF showing how V(s) evolves over time.

    :param num_episodes: Number of episodes agent trains for, defaults to 500
    :param make_gif: Make a gif of values and policy converging, defaults to True
    :param gif_filename: Visualisation GIF filename, defaults to "convergence.gif"
    :return: Trained agent
    """
    env = GridworldEnv(discount=0.9)
    agent = QLearningAgent(rows=5, cols=5, num_actions=4, alpha=0.1, gamma=0.9, epsilon=0.1)

    frames = []  # to store PIL images for each episode
    for episode in range(num_episodes):
        state = env.reset()
        done = False
        
        # 100 steps per episode
        for _ in range(1000):
            action = agent.get_action(state)
            next_state, reward, done, _ = env.step(action)
            agent.update(state, action, reward, next_state)
            state = next_state

        # Compute value function and policy
        values = _compute_value_function(agent.Q)
        policy = _compute_greedy_policy(agent.Q)
        
        if make_gif:
            fig = plot_values_and_policy(values, policy,
                                         title_values=f"Values (Episode {episode+1})",
                                         title_policy=f"Policy (Episode {episode+1})")
            pil_img = fig_to_pil(fig)
            frames.append(pil_img)
            fig.clf()

    if make_gif and len(frames) > 1:
        frames[0].save(
            gif_filename,
            save_all=True,
            append_images=frames[1:],
            duration=150,
            loop=0
        )
        print(f"GIF saved as {gif_filename}")

    # Return the trained agent (with Q) so we can do final plotting
    return agent


def _compute_value_function(Q):
    """
    V(s) = max_a Q(s,a). 

    :param Q: Q table of agent
    :return: State values
    """
    values = np.max(Q, axis=2)
    return values


def _compute_greedy_policy(Q):
    """
    Return a 2D array where each cell is a list of the argmax actions.
    If there's a tie, list multiple actions.

    :param Q: Q table of agent
    :return: policy of agent
    """
    rows, cols, num_actions = Q.shape
    policy = np.empty((rows, cols), dtype=object)
    
    for r in range(rows):
        for c in range(cols):
            q_values = Q[r, c, :]
            max_q = np.max(q_values)
            best_actions = [a for a in range(num_actions) if q_values[a] == max_q]
            policy[r, c] = best_actions
    return policy
