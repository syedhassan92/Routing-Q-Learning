import tkinter as tk
import math
import threading
import queue
from typing import List, Tuple, Dict, Set, Optional, Any
import networkx as nx


class DijkstraVisualizer:
    """Visualizes Dijkstra's algorithm step-by-step on canvas."""

    def __init__(self, canvas: tk.Canvas, nodes: Dict[int, Tuple[int, int]], 
                 edges: Dict[Tuple[int, int], float], G: nx.DiGraph, 
                 source: int, target: int):
        self.canvas = canvas
        self.nodes = nodes
        self.edges = edges
        self.G = G
        self.source = source
        self.target = target
        self.explored = set()  # Explored nodes
        self.frontier = set()  # Active frontier nodes
        self.distances = {n: float('inf') for n in self.nodes}
        self.distances[source] = 0
        self.parent = {}  # For path reconstruction
        self.step_log = []  # Log of algorithm steps

    def run_step_by_step(self, step_callback=None):
        import heapq

        pq = [(0, self.source)]
        self.explored = set()
        self.frontier = {self.source}

        while pq:
            dist, u = heapq.heappop(pq)

            if u in self.explored:
                continue

            self.explored.add(u)
            self.frontier.discard(u)

            # Log step
            step_info = {
                'current': u,
                'distance': dist,
                'explored': self.explored.copy(),
                'frontier': self.frontier.copy(),
                'distances': self.distances.copy()
            }
            self.step_log.append(step_info)

            if step_callback:
                step_callback(step_info)

            if u == self.target:
                break

            for v in self.G.neighbors(u):
                if v not in self.explored:
                    new_dist = dist + self.G[u][v]['weight']
                    if new_dist < self.distances[v]:
                        self.distances[v] = new_dist
                        self.parent[v] = u
                        self.frontier.add(v)
                        heapq.heappush(pq, (new_dist, v))

        return self.step_log


class QLearningRolloutVisualizer:

    def __init__(
        self,
        agent: Any,
        G: nx.DiGraph,
        source: int,
        target: int,
        max_steps: int = 50,
    ):
        self.agent = agent
        self.G = G
        self.source = source
        self.target = target
        self.max_steps = max_steps

        self.visited: Set[int] = set()
        self.current: Optional[int] = None
        self.next_node: Optional[int] = None
        self.step_log: List[Dict] = []

    def run_step_by_step(self, step_callback=None):
        self.step_log = []

        state = self.source
        visited: Set[int] = {state}
        path: List[int] = [state]
        cost = 0.0

        # Initialize "current" state for the first render.
        self.current = state
        self.next_node = None
        self.visited = visited.copy()

        for _ in range(self.max_steps):
            if state == self.target:
                break

            neighbours = list(self.G.neighbors(state))
            if not neighbours:
                step_info = {
                    "current": state,
                    "neighbors": [],
                    "q_values": {},
                    "chosen": None,
                    "edge_weight": None,
                    "cumulative_cost": cost,
                    "path": path.copy(),
                    "visited": visited.copy(),
                    "epsilon": float(getattr(self.agent, "epsilon", 0.0)),
                    "done": True,
                    "reason": "dead_end",
                }
                self.step_log.append(step_info)
                if step_callback:
                    step_callback(step_info)
                break

            q_map: Dict[int, float] = {}
            scored: List[Tuple[float, int]] = []
            for n in neighbours:
                try:
                    qv = float(self.agent.q_table[state, n])
                except Exception:
                    continue
                q_map[n] = qv
                scored.append((qv, n))

            if not scored:
                step_info = {
                    "current": state,
                    "neighbors": neighbours,
                    "q_values": {},
                    "chosen": None,
                    "edge_weight": None,
                    "cumulative_cost": cost,
                    "path": path.copy(),
                    "visited": visited.copy(),
                    "epsilon": float(getattr(self.agent, "epsilon", 0.0)),
                    "done": True,
                    "reason": "q_table_unavailable",
                }
                self.step_log.append(step_info)
                if step_callback:
                    step_callback(step_info)
                break

            _, nxt = max(scored, key=lambda t: t[0])
            w = float(self.G[state][nxt].get("weight", 0.0))
            new_cost = cost + w

            step_info = {
                "current": state,
                "neighbors": neighbours,
                "q_values": q_map,
                "chosen": nxt,
                "edge_weight": w,
                "cumulative_cost": new_cost,
                "path": (path + [nxt]),
                "visited": visited.copy(),
                "epsilon": float(getattr(self.agent, "epsilon", 0.0)),
                "done": (nxt == self.target),
                "reason": ("reached_target" if nxt == self.target else "step"),
            }
            self.step_log.append(step_info)
            if step_callback:
                step_callback(step_info)

            visited.add(nxt)
            path.append(nxt)
            state = nxt
            cost = new_cost

        return self.step_log


class QLearningVisualizer:

    def __init__(self):
        self.episode_rewards = []
        self.q_values_history = []
        self.best_actions = {}
        self.convergence_data = {}

    def track_episode(self, episode: int, total_reward: float, q_table, agent):
        self.episode_rewards.append({
            'episode': episode,
            'reward': total_reward,
            'epsilon': agent.epsilon
        })

        # Track average Q-value
        import numpy as np
        avg_q = np.mean(q_table[q_table != 0]) if np.any(q_table) else 0
        self.convergence_data[episode] = {
            'avg_q': avg_q,
            'max_q': np.max(q_table),
            'epsilon': agent.epsilon
        }

    def get_convergence_summary(self):
        if not self.episode_rewards:
            return {}

        rewards = [r['reward'] for r in self.episode_rewards]
        recent_avg = sum(rewards[-100:]) / len(rewards[-100:]) if len(rewards) >= 100 else sum(rewards) / len(rewards)

        return {
            'total_episodes': len(self.episode_rewards),
            'final_reward': rewards[-1],
            'average_recent': recent_avg,
            'max_reward': max(rewards),
            'min_reward': min(rewards)
        }


class PacketAnimator:

    def __init__(self, canvas: tk.Canvas, nodes: Dict[int, Tuple[int, int]], 
                 edges: Dict[Tuple[int, int], float]):
        self.canvas = canvas
        self.nodes = nodes
        self.edges = edges
        self.packets = []  # List of animated packets
        self.packet_id_counter = 0

    def create_packet(self, path: List[int], color: str = "#ef4444", 
                      speed: float = 2.0) -> int:
        packet_id = self.packet_id_counter
        self.packet_id_counter += 1

        packet = {
            'id': packet_id,
            'path': path,
            'color': color,
            'speed': speed,
            'current_segment': 0,  # Which edge in path
            'progress': 0.0,  # 0-1 progress along current edge
            'total_cost': 0.0,
            'visited_nodes': [path[0]],
            'canvas_obj': None
        }
        self.packets.append(packet)
        return packet_id

    def animate_packet(self, packet_id: int) -> bool:
        packet = next((p for p in self.packets if p['id'] == packet_id), None)
        if not packet:
            return False

        path = packet['path']
        segment = packet['current_segment']

        if segment >= len(path) - 1:
            # Reached destination
            return False

        u, v = path[segment], path[segment + 1]
        x1, y1 = self.nodes[u]
        x2, y2 = self.nodes[v]

        dx, dy = x2 - x1, y2 - y1
        dist = math.hypot(dx, dy)

        if dist == 0:
            packet['current_segment'] += 1
            return True

        # Move along edge
        packet['progress'] += packet['speed'] / dist

        if packet['progress'] >= 1.0:
            # Reached next node
            packet['progress'] = 0.0
            packet['current_segment'] += 1
            packet['visited_nodes'].append(v)

            # Add edge cost
            if (u, v) in self.edges:
                packet['total_cost'] += self.edges[(u, v)]

            if packet['current_segment'] >= len(path) - 1:
                return False  # Destination reached

        return True

    def get_packet_position(self, packet_id: int) -> Tuple[float, float]:
        packet = next((p for p in self.packets if p['id'] == packet_id), None)
        if not packet:
            return None

        path = packet['path']
        segment = packet['current_segment']

        if segment >= len(path) - 1:
            return self.nodes[path[-1]]

        u, v = path[segment], path[segment + 1]
        x1, y1 = self.nodes[u]
        x2, y2 = self.nodes[v]

        x = x1 + (x2 - x1) * packet['progress']
        y = y1 + (y2 - y1) * packet['progress']

        return (x, y)

    def get_packet_stats(self, packet_id: int):
        packet = next((p for p in self.packets if p['id'] == packet_id), None)
        if not packet:
            return {}

        return {
            'path': packet['path'],
            'total_cost': packet['total_cost'],
            'visited_nodes': packet['visited_nodes'],
            'progress': packet['progress'],
            'segment': packet['current_segment']
        }


class CostComputationVisualizer:

    def __init__(self, edges: Dict[Tuple[int, int], float]):
        self.edges = edges
        self.cost_breakdown = []

    def record_hop(self, from_node: int, to_node: int, edge_weight: float, cumulative_cost: float):
        self.cost_breakdown.append({
            'from': from_node,
            'to': to_node,
            'hop_cost': edge_weight,
            'cumulative_cost': cumulative_cost,
            'hop_number': len(self.cost_breakdown) + 1
        })

    def get_cost_table(self):
        return self.cost_breakdown

    def get_total_cost(self) -> float:
        return self.cost_breakdown[-1]['cumulative_cost'] if self.cost_breakdown else 0.0

    def format_for_display(self):
        if not self.cost_breakdown:
            return "No hops recorded"

        lines = ["HOP ANALYSIS", "=" * 50]
        lines.append(f"{'Hop':<6} {'From':<6} {'To':<6} {'Cost':<8} {'Cumulative':<12}")
        lines.append("-" * 50)

        for hop in self.cost_breakdown:
            lines.append(
                f"{hop['hop_number']:<6} {hop['from']:<6} {hop['to']:<6} "
                f"{hop['hop_cost']:<8.1f} {hop['cumulative_cost']:<12.1f}"
            )

        lines.append("-" * 50)
        lines.append(f"TOTAL COST: {self.get_total_cost():.1f}")

        return "\n".join(lines)


class ComparisonVisualizer:

    def __init__(self):
        self.dijkstra_steps = []
        self.ql_steps = []
        self.dijkstra_path = None
        self.ql_path = None
        self.dijkstra_cost = None
        self.ql_cost = None

    def set_dijkstra_result(self, path: List[int], cost: float, steps: List[Dict]):
        self.dijkstra_path = path
        self.dijkstra_cost = cost
        self.dijkstra_steps = steps

    def set_ql_result(self, path: List[int], cost: float):
        self.ql_path = path
        self.ql_cost = cost

    def generate_comparison_report(self):
        lines = []
        lines.append("\n" + "=" * 60)
        lines.append("ALGORITHM COMPARISON REPORT")
        lines.append("=" * 60)

        if self.dijkstra_path:
            lines.append("\n[DIJKSTRA'S ALGORITHM]")
            lines.append(f"  Path: {' -> '.join(map(str, self.dijkstra_path))}")
            lines.append(f"  Cost: {self.dijkstra_cost}")
            lines.append(f"  Hops: {len(self.dijkstra_path) - 1}")
            lines.append(f"  Exploration Steps: {len(self.dijkstra_steps)}")

        if self.ql_path:
            lines.append("\n[Q-LEARNING AGENT]")
            lines.append(f"  Path: {' -> '.join(map(str, self.ql_path))}")
            lines.append(f"  Cost: {self.ql_cost}")
            lines.append(f"  Hops: {len(self.ql_path) - 1}")

        if self.dijkstra_path and self.ql_path:
            lines.append("\n[COMPARISON]")
            if self.dijkstra_cost == self.ql_cost:
                lines.append("  ✓ Both found OPTIMAL path with same cost")
            elif self.dijkstra_cost < self.ql_cost:
                diff = self.ql_cost - self.dijkstra_cost
                lines.append(f"  ✓ Dijkstra found cheaper path (saves {diff:.1f})")
            else:
                diff = self.dijkstra_cost - self.ql_cost
                lines.append(f"  ✓ Q-Learning found cheaper path (saves {diff:.1f})")

            if len(self.dijkstra_path) == len(self.ql_path):
                lines.append("  ✓ Both paths have same hop count")
            elif len(self.dijkstra_path) < len(self.ql_path):
                diff = len(self.ql_path) - len(self.dijkstra_path)
                lines.append(f"  ✓ Dijkstra uses fewer hops ({diff} fewer)")
            else:
                diff = len(self.dijkstra_path) - len(self.ql_path)
                lines.append(f"  ✓ Q-Learning uses fewer hops ({diff} fewer)")

        lines.append("=" * 60)

        return "\n".join(lines)
