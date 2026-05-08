"""
q_learning_agent.py
====================
A tabular Q-Learning agent that learns optimal routing paths in a weighted
directed network graph.  The agent is trained from scratch -- no ML frameworks.
"""

from typing import List, Tuple

import numpy as np
import networkx as nx


class QLearningAgent:
    """Tabular Q-Learning agent for network routing.

    Attributes
    ----------
    n_states : int
        Number of nodes (states) in the network.
    q_table : np.ndarray
        Q-value matrix of shape ``(n_states, n_states)``.
    alpha : float
        Learning rate.
    gamma : float
        Discount factor.
    epsilon : float
        Current exploration rate (decays over training).
    """

    def __init__(
        self,
        n_states: int,
        n_actions: int,
        alpha: float = 0.2,
        gamma: float = 0.9,
        epsilon: float = 0.3,
    ) -> None:
        """Initialise the Q-Learning agent.

        Parameters
        ----------
        n_states : int
            Total number of nodes in the graph.
        n_actions : int
            Same as *n_states* (action = next-node to move to).
        alpha : float
            Learning rate  (default 0.2).
        gamma : float
            Discount factor (default 0.9).
        epsilon : float
            Initial exploration rate (default 0.3).
        """
        self.n_states = n_states
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.q_table = np.zeros((n_states, n_states))

    def choose_action(self, state: int, valid_actions: List[int]) -> int:
        """Select an action using an epsilon-greedy policy.

        Parameters
        ----------
        state : int
            Current node.
        valid_actions : list[int]
            Reachable neighbours the agent can move to.

        Returns
        -------
        int
            The chosen next node.
        """
        if np.random.random() < self.epsilon:
            return int(np.random.choice(valid_actions))
        # Greedy: pick the valid action with the highest Q-value
        q_values = self.q_table[state, valid_actions]
        best_idx = int(np.argmax(q_values))
        return valid_actions[best_idx]

    def update(self, state: int, action: int, reward: float, next_state: int) -> None:
        """Apply the Q-learning update rule.

        Q(s, a) <- Q(s, a) + alpha [ r + gamma * max_a' Q(s', a') - Q(s, a) ]

        Parameters
        ----------
        state : int
            State before the action.
        action : int
            Action taken (next node).
        reward : float
            Immediate reward received.
        next_state : int
            State after the action.
        """
        best_next = np.max(self.q_table[next_state])
        td_target = reward + self.gamma * best_next
        td_error = td_target - self.q_table[state, action]
        self.q_table[state, action] += self.alpha * td_error


# -- Reward helpers ------------------------------------------------------------

def compute_reward(
    G: nx.DiGraph,
    current: int,
    next_node: int,
    target: int,
    visited: set,
) -> float:
    """Calculate the immediate reward for moving from *current* to *next_node*.

    Reward scheme
    -------------
    * **+100** for reaching the destination.
    * **-weight** of the traversed link (penalises costly edges).
    * **-1** step penalty  (encourages shorter paths).
    * **-10** if *next_node* was already visited (loop penalty).

    Parameters
    ----------
    G : nx.DiGraph
        The network graph.
    current : int
        Current node.
    next_node : int
        Node the agent moves to.
    target : int
        Destination node.
    visited : set
        Set of already-visited nodes in the current episode.

    Returns
    -------
    float
        The computed reward.
    """
    reward = 0.0

    # Goal bonus
    if next_node == target:
        reward += 100.0

    # Link-cost penalty
    weight = G[current][next_node]["weight"]
    reward -= weight

    # Step penalty
    reward -= 1.0

    # Loop penalty
    if next_node in visited:
        reward -= 10.0

    return reward


# -- Training loop -------------------------------------------------------------

def train(
    agent: QLearningAgent,
    G: nx.DiGraph,
    source: int,
    target: int,
    episodes: int = 1000,
    max_steps: int = 50,
    epsilon_decay: float = 0.995,
    epsilon_min: float = 0.05,
    verbose: bool = True,
) -> List[float]:
    """Train the Q-learning agent on the directed network.

    Parameters
    ----------
    agent : QLearningAgent
        The agent to train.
    G : nx.DiGraph
        Directed network graph.
    source : int
        Start node.
    target : int
        Destination node.
    episodes : int
        Number of training episodes (default 1000).
    max_steps : int
        Maximum steps per episode before truncation (default 50).
    epsilon_decay : float
        Multiplicative decay applied to epsilon each episode.
    epsilon_min : float
        Minimum value of epsilon.
    verbose : bool
        If *True*, print progress every 100 episodes.

    Returns
    -------
    list[float]
        Per-episode total rewards (learning curve).
    """
    episode_rewards: List[float] = []

    for ep in range(1, episodes + 1):
        state = source
        visited = {state}
        total_reward = 0.0

        for _ in range(max_steps):
            # For DiGraph, neighbors() returns successors (outgoing edges)
            neighbours = list(G.neighbors(state))
            if not neighbours:
                break  # dead-end node
            action = agent.choose_action(state, neighbours)

            reward = compute_reward(G, state, action, target, visited)
            agent.update(state, action, reward, action)

            total_reward += reward
            visited.add(action)
            state = action

            if state == target:
                break

        episode_rewards.append(total_reward)

        # Decay epsilon
        agent.epsilon = max(epsilon_min, agent.epsilon * epsilon_decay)

        if verbose and ep % 100 == 0:
            avg = np.mean(episode_rewards[-100:])
            print(f"Episode {ep:>5d}: Avg Reward = {avg:.1f}")

    return episode_rewards


# -- Path extraction -----------------------------------------------------------

def extract_path(
    agent: QLearningAgent,
    source: int,
    target: int,
    G: nx.DiGraph,
    max_steps: int = 50,
) -> Tuple[List[int], int]:
    """Extract the greedy path from the trained Q-table.

    Parameters
    ----------
    agent : QLearningAgent
        A trained agent.
    source : int
        Start node.
    target : int
        Destination node.
    G : nx.DiGraph
        Directed network graph (used for neighbour lookup & edge weights).
    max_steps : int
        Safety limit to avoid infinite loops.

    Returns
    -------
    tuple[list[int], int]
        (path, cost) -- the greedy path and its total edge-weight cost.
    """
    path = [source]
    state = source
    cost = 0

    for _ in range(max_steps):
        if state == target:
            break
        neighbours = list(G.neighbors(state))
        if not neighbours:
            break  # dead-end
        q_values = agent.q_table[state, neighbours]
        best_idx = int(np.argmax(q_values))
        next_node = neighbours[best_idx]
        cost += G[state][next_node]["weight"]
        path.append(next_node)
        state = next_node

    return path, cost


if __name__ == "__main__":
    from dijkstra_routing import format_path
    import networkx as nx

    # Create a simple graph for testing since network_graph is missing
    G = nx.Graph()
    G.add_edges_from([
        (0, 1, {"weight": 10}), (1, 2, {"weight": 15}),
        (2, 3, {"weight": 10}), (3, 4, {"weight": 20}),
        (4, 5, {"weight": 10}), (2, 5, {"weight": 50})
    ])

    n = max(G.nodes) + 1
    agent = QLearningAgent(n, n)
    rewards = train(agent, G, source=2, target=5, episodes=1000)
    path, cost = extract_path(agent, 2, 5, G)
    print(f"\nQ-Learning Path: [{format_path(path)}], Cost: {cost}")
