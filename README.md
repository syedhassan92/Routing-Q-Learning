# Intelligent Network Routing using Reinforcement Learning

A Python simulation that compares **Q-Learning-based routing** against **Dijkstra's Algorithm** on a weighted graph representing a computer network.

## Project Structure

| File | Description |
|------|-------------|
| `network_graph.py` | Builds and visualizes the 6-router network topology |
| `dijkstra_routing.py` | Shortest-path routing via Dijkstra's algorithm |
| `q_learning_agent.py` | Tabular Q-Learning agent with training loop |
| `dynamic_network.py` | Simulates link congestion and re-evaluates routes |
| `performance_comparison.py` | Prints formatted comparison tables |
| `visualizations.py` | Matplotlib plots (learning curve, cost bars, path overlay) |
| `main.py` | Orchestrates the full simulation pipeline |

## How to Run

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the simulation

```bash
python main.py
```

## What It Does

1. **Builds** a weighted undirected graph with 8 routers and 11 links
2. **Visualizes** the network topology
3. **Runs Dijkstra** to find the optimal shortest path
4. **Trains a Q-Learning agent** (1000 episodes) to discover routing paths
5. **Simulates congestion** by increasing a link's weight
6. **Re-runs both algorithms** on the modified network
7. **Prints a comparison table** and generates plots

## Sample Output

```
============================================================
  Intelligent Network Routing using Reinforcement Learning
============================================================

Network: 8 routers, 11 links
Source : 0  |  Destination : 7

Dijkstra Path : [0->1->4->7], Cost: 6
Training Q-Learning agent ...

Episode   100: Avg Reward = -23.4
Episode   200: Avg Reward = 78.5
...
Episode  1000: Avg Reward = 93.2

Q-Learning Path: [0->1->4->7], Cost: 6

  SIMULATING CONGESTION
  Congestion simulated on link (4-7): weight 3 -> 20

==============================================
       ROUTING COMPARISON TABLE
==============================================
Strategy       | Path                 | Cost | Hops
---------------+----------------------+------+-----
Dijkstra       | 0->1->4->7           |    6 |    3
Q-Learning     | 0->1->4->7           |    6 |    3
Dijkstra*      | 0->1->3->6->7        |    9 |    4  <- after congestion
Q-Learning*    | 0->1->3->6->7        |    9 |    4  <- adapts!
==============================================
```

## Visualizations

The simulation produces four plots:

1. **Learning Curve** - Reward vs Episode (shows agent convergence)
2. **Cost Comparison Bar Chart** - Dijkstra vs Q-Learning (before & after congestion)
3. **Path Overlay (Pre-Congestion)** - Blue = Dijkstra, Red dashed = Q-Learning
4. **Path Overlay (Post-Congestion)** - Updated paths on modified network

## References

- Q-Learning for Network Routing: https://ieeexplore.ieee.org/document/8701570
- NetworkX Documentation: https://networkx.org/documentation/stable/
- Matplotlib Documentation: https://matplotlib.org/stable/users/index.html
