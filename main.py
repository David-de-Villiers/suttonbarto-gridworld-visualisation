from agent import QLearningAgent
import numpy as np
from train import train_q_learning
from visualisation import plot_values_and_policy, fig_to_pil

def _compute_greedy_policy(Q):
    """
    Compute Greedy Policy of trained agent

    :param Q: Q-table of trained agent
    :return: Greedy policy using best actions
    """
    rows, cols, num_actions = Q.shape
    policy = np.empty((rows, cols), dtype=object)

    for y in range(rows):
        for x in range(cols):
            q_vals = Q[y, x, :]
            max_q = np.max(q_vals)
            best_actions = [a for a in range(num_actions) if q_vals[a] == max_q]
            policy[y, x] = best_actions

    return policy


def main():
    agent = train_q_learning(
        num_episodes=800,
        make_gif=True,
        gif_filename="convergence.gif"
    )
    print("Training finished.")

    final_values = agent.Q.max(axis=2)
    final_policy = _compute_greedy_policy(agent.Q)

    # Create figures
    fig = plot_values_and_policy(final_values, final_policy,
                                 title_values="Final Values",
                                 title_policy="Final Policy")
    pil_img = fig_to_pil(fig)
    pil_img.save("final_values_and_policy.png")

    print("Saved final policy image as final_values_and_policy.png")

if __name__ == "__main__":
    main()
