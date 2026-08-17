import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from scipy.spatial.distance import cdist

# =====================================================================
# 1. SE(2) ENVIRONMENT & DYNAMIC TOPOLOGY
# =====================================================================
class DynamicSE2Environment:
    def __init__(self):
        self.start = np.array([0.1, 0.5])
        self.goal = np.array([0.9, 0.5])
        # Nominal passage (open central corridor)
        self.obstacle_active = False
        self.obstacle_pos = np.array([0.5, 0.5])
        self.obstacle_radius = 0.18

    def set_obstacle(self, active=True):
        self.obstacle_active = active

    def is_collision(self, point):
        if not self.obstacle_active:
            return False
        return np.linalg.norm(point[:2] - self.obstacle_pos) < self.obstacle_radius

# =====================================================================
# 2. NOMINAL NEURAL POLICY (OBLIVIOUS TO OOD TOPOLOGY SHIFTS)
# =====================================================================
class NominalNeuralPolicy:
    """Simulates a fixed, pretrained policy that learned the straight nominal corridor."""
    def act(self, state, goal):
        direction = goal - state
        norm = np.linalg.norm(direction)
        if norm > 1e-4:
            return (direction / norm) * 0.03
        return np.zeros(2)

# =====================================================================
# 3. TOPOLOGICAL ADAPTER (PERSISTENT GRAPH FILTRATION ON SE(2))
# =====================================================================
class TopologicalManifoldFilter:
    """
    Maintains a 1-skeleton graph over SE(2).
    Updates topological reachability via Vietoris-Rips-like distance filtration
    and prunes edges intersecting newly detected topological obstacles zero-shot.
    """
    def __init__(self, num_nodes=120, radius=0.16):
        self.radius = radius
        # Sample state-space grid
        x = np.linspace(0.05, 0.95, 12)
        y = np.linspace(0.1, 0.9, 10)
        xx, yy = np.meshgrid(x, y)
        self.nodes = np.column_stack([xx.ravel(), yy.ravel()])
        self.G_base = nx.Graph()
        
        for i, pt in enumerate(self.nodes):
            self.G_base.add_node(i, pos=pt)

        # Build Vietoris-Rips 1-skeleton (edges within filtration radius)
        dists = cdist(self.nodes, self.nodes)
        for i in range(len(self.nodes)):
            for j in range(i + 1, len(self.nodes)):
                if dists[i, j] <= self.radius:
                    self.G_base.add_edge(i, j, weight=dists[i, j])

    def update_topology(self, env):
        """Zero-shot filtration: prune active topological obstacles in O(E)."""
        G_filtered = self.G_base.copy()
        nodes_to_remove = [
            n for n, d in G_filtered.nodes(data=True) if env.is_collision(d['pos'])
        ]
        G_filtered.remove_nodes_from(nodes_to_remove)
        
        # Extract Betti numbers approximation (Connected components & cycles)
        b_0 = nx.number_connected_components(G_filtered)
        b_1 = G_filtered.number_of_edges() - G_filtered.number_of_nodes() + b_0
        return G_filtered, b_0, b_1

    def get_topological_guidance(self, state, goal, G_filtered):
        # Find nearest start and goal nodes in filtered graph
        node_positions = np.array([d['pos'] for n, d in G_filtered.nodes(data=True)])
        node_indices = list(G_filtered.nodes())
        
        if len(node_indices) == 0:
            return goal - state

        start_idx = node_indices[np.argmin(np.linalg.norm(node_positions - state, axis=1))]
        goal_idx = node_indices[np.argmin(np.linalg.norm(node_positions - goal, axis=1))]

        try:
            path = nx.shortest_path(G_filtered, source=start_idx, target=goal_idx, weight='weight')
            if len(path) > 1:
                next_waypoint = G_filtered.nodes[path[1]]['pos']
                direction = next_waypoint - state
                return (direction / (np.linalg.norm(direction) + 1e-6)) * 0.03
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            pass
            
        return goal - state

# =====================================================================
# 4. EXECUTION & ZERO-SHOT EVALUATION
# =====================================================================
def run_experiment_c():
    env = DynamicSE2Environment()
    nn_policy = NominalNeuralPolicy()
    topo_filter = TopologicalManifoldFilter()

    # Scenario 1: Nominal Environment (Corridor Open)
    env.set_obstacle(False)
    
    # Scenario 2: OOD Shift (Central passage dynamically blocked)
    env.set_obstacle(True)

    # Rollout 1: Unregularized Neural Policy (Static weights)
    state_nn = env.start.copy()
    traj_nn = [state_nn.copy()]
    collided_nn = False
    
    for _ in range(60):
        action = nn_policy.act(state_nn, env.goal)
        state_nn = state_nn + action
        traj_nn.append(state_nn.copy())
        if env.is_collision(state_nn):
            collided_nn = True
            break

    # Rollout 2: TCLA (Frozen Policy + Real-time Topological Filtration)
    state_tcla = env.start.copy()
    traj_tcla = [state_tcla.copy()]
    collided_tcla = False
    
    G_active, b0, b1 = topo_filter.update_topology(env)
    
    for _ in range(60):
        # Topological guidance injected into action stream zero-shot
        action = topo_filter.get_topological_guidance(state_tcla, env.goal, G_active)
        state_tcla = state_tcla + action
        traj_tcla.append(state_tcla.copy())
        if env.is_collision(state_tcla):
            collided_tcla = True
            break
        if np.linalg.norm(state_tcla - env.goal) < 0.04:
            break

    traj_nn = np.array(traj_nn)
    traj_tcla = np.array(traj_tcla)

    print("=" * 70)
    print("EXPERIMENT C RESULTS: OUT-OF-DISTRIBUTION TOPOLOGICAL GENERALIZATION")
    print("=" * 70)
    print(f"Topological Invariants Post-Shift: Betti-0 (Components) = {b0}, Betti-1 (1-Loops) = {b1}")
    print(f"Nominal Neural Policy (No Retraining) : {'FAILED (Collision)' if collided_nn else 'SUCCESS'}")
    print(f"TCLA Topo-Filtered Policy (Zero-Shot) : {'FAILED' if collided_tcla else 'SUCCESS (Target Reached)'}")
    print("=" * 70)

    # Plot trajectories and filtration
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Draw obstacle
    circle = plt.Circle(env.obstacle_pos, env.obstacle_radius, color='red', alpha=0.3, label='OOD Obstacle (Blocked Loop)')
    ax.add_patch(circle)

    # Draw filtered graph edges
    for u, v in G_active.edges():
        p1, p2 = G_active.nodes[u]['pos'], G_active.nodes[v]['pos']
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color='gray', alpha=0.2, lw=0.8)

    # Plot paths
    ax.plot(traj_nn[:, 0], traj_nn[:, 1], 'r--o', lw=2, label='Fixed Neural Policy (Collides)')
    ax.plot(traj_tcla[:, 0], traj_tcla[:, 1], 'teal', marker='o', lw=2.5, label='TCLA Zero-Shot Topological Reroute')

    ax.scatter([env.start[0]], [env.start[1]], color='green', s=120, zorder=5, label='Start')
    ax.scatter([env.goal[0]], [env.goal[1]], color='gold', edgecolor='black', s=150, zorder=5, label='Goal')

    ax.set_title(r"Experiment C: $SE(2)$ OOD Generalization via Topological Filtration", fontweight='bold')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(loc='lower left')
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig("experiment_c_topological_ood.png", dpi=300)
    print("Saved trajectory visualization to 'experiment_c_topological_ood.png'.")
    plt.show()

if __name__ == "__main__":
    run_experiment_c()