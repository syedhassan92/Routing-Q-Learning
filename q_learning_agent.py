from typing import List, Tuple

import numpy as np
import networkx as nx


class QLearningAgent:
    def __init__(self,n_states: int,n_actions: int,alpha: float = 0.2,gamma: float = 0.9,epsilon: float = 0.3,) -> None:
        self.n_states = n_states
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.q_table = np.zeros((n_states, n_states))

    def choose_action(self, state: int, valid_actions: List[int]) -> int:
        if np.random.random() < self.epsilon:
            return int(np.random.choice(valid_actions))
        # Greedy: pick the valid action with the highest Q-value
        q_values = self.q_table[state, valid_actions]
        best_idx = int(np.argmax(q_values))
        return valid_actions[best_idx]

    def update(self, state: int, action: int, reward: float, next_state: int) -> None:

        #Q(s, a) ← Q(s, a) + α[r + γ·max_a'(Q(s', a')) - Q(s, a)]

        best_next = np.max(self.q_table[next_state])
        td_target = reward + self.gamma * best_next
        td_error = td_target - self.q_table[state, action]
        self.q_table[state, action] += self.alpha * td_error


# -- Reward helpers ------------------------------------------------------------

def compute_reward(G: nx.DiGraph,current: int,next_node: int,target: int,visited: set,) -> float:
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

def train(agent: QLearningAgent,G: nx.DiGraph,source: int,target: int,episodes: int = 1000,max_steps: int = 50,epsilon_decay: float = 0.995,epsilon_min: float = 0.05,verbose: bool = True,) -> List[float]:
   
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

def extract_path(agent: QLearningAgent,source: int,target: int,G: nx.DiGraph,max_steps: int = 50,) -> Tuple[List[int], int]:
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
    from dijkstra_routing import dijkstra_path, format_path
    import networkx as nx

    G = nx.Graph()
    # Path 1: 4 hops, total weight = 9.9 (Dijkstra's favourite because 9.9 < 10)
    G.add_edges_from([
        (0, 1, {"weight": 2.0}), 
        (1, 2, {"weight": 2.0}),
        (2, 3, {"weight": 2.9}), 
        (3, 4, {"weight": 3.0}),
    ])
    
    # Path 2: 2 hops, total weight = 10.0 (Q-Learning's favourite because fewer step penalties)
    G.add_edges_from([
        (0, 5, {"weight": 5.0}), 
        (5, 4, {"weight": 5.0})
    ])

    print("--- Evaluating Paths from Node 0 to Node 4 ---")

    # Dijkstra
    d_path, d_cost = dijkstra_path(G, 0, 4)
    print(f"\nDijkstra Path:   [{format_path(d_path)}], Cost (Weight): {d_cost:.1f}, Hops: {len(d_path)-1}")

    # Q-Learning
    n = max(G.nodes) + 1
    agent = QLearningAgent(n, n)
    rewards = train(agent, G, source=0, target=4, episodes=1000, verbose=False)
    q_path, q_cost = extract_path(agent, 0, 4, G)
    print(f"Q-Learning Path: [{format_path(q_path)}], Cost (Weight): {q_cost:.1f}, Hops: {len(q_path)-1}\n")

