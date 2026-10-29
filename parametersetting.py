def load_signed_graph(file_path, prob_mode="uniform"):
    G         = nx.DiGraph()
    edge_sign = {}
    raw_edges = []

    with open(file_path, "r") as f:
        for line in f:
            if line.startswith("%") or not line.strip():
                continue
            parts = line.strip().split(",")
            u, v, sign = int(parts[0]), int(parts[1]), int(parts[2])
            G.add_edge(u, v)
            edge_sign[(u, v)] = sign
            raw_edges.append((u, v, sign))

    probabilities = {}
    for u, v, sign in raw_edges:
        if prob_mode == "indegree":
            in_deg = G.in_degree(v)
            probabilities[(u, v)] = 1.0 / in_deg if in_deg > 0 else 1.0
        elif prob_mode == "weighted":
            probabilities[(u, v)] = min(abs(sign), 1.0)
        else:
            probabilities[(u, v)] = random.uniform(0, 1)

    print(f"Loaded   : {G.number_of_nodes()} nodes, "
          f"{G.number_of_edges()} edges")
    print(f"Prob mode: {prob_mode}")
    return G, probabilities, edge_sign

def detect_communities(G):
    G_und = G.to_undirected()
    partition = community_louvain.best_partition(G_und, random_state=42)
    comm_count = len(set(partition.values()))
    print(f"[Topology] Louvain detected {comm_count} modular communities.")
    return partition


def initialize_parameters(G, probabilities):
    z = {n: random.uniform(0.5, 1.0) for n in G.nodes()}
    thresh = {n: random.uniform(0.01, 0.05) for n in G.nodes()}
    a = {e: random.uniform(0.0, 0.3) for e in probabilities}
    b = {e: random.uniform(0.3, 0.7) for e in probabilities}
    c_pos = {e: random.uniform(0.5, 1.0) for e in probabilities}
    r_neg = {e: random.uniform(0.5, 1.0) for e in probabilities}
    return z, a, b, thresh, c_pos, r_neg
def balance_factor(u, v, edge_sign, opinions):
    lu      = opinions[u]
    sign_uv = edge_sign[(u, v)]
    if lu >= 0 and sign_uv == 1:
        return 1.0
    if lu < 0  and sign_uv == -1:
        return 0.5
    if lu >= 0 and sign_uv == -1:
        return 0.5
    return 0.5
ALPHA_GRID = (0.2, 0.5, 0.7, 1.0)
BETA_GRID  = (0.1, 0.3, 0.5, 0.7)

def sweep_alpha_beta(algo, k, common, obj, thresh, algo_kwargs=None,
                     alpha_grid=ALPHA_GRID, beta_grid=BETA_GRID):
    rows = []
    for alpha, beta in itertools.product(alpha_grid, beta_grid):
        kw = {**common, **obj, **(algo_kwargs or {}),
              "k": k, "thresh": thresh, "alpha": alpha, "beta": beta}
        S, log = algo(**kw)
        sp = log[-1][3] if log else np.nan      
        sn = log[-1][4] if log else np.nan
        rows.append(dict(alpha=alpha, beta=beta, seeds=tuple(sorted(S)),
                         sigma_pos=sp, sigma_neg=sn))
    return pd.DataFrame(rows)


def recompute_F(sp, sn, xi, n_nodes, alpha, beta):
    return alpha * sp / n_nodes + (1 - alpha) * xi - beta * sn / n_nodes


def recompute_grid(sp, sn, xi, n_nodes,
                   alpha_grid=ALPHA_GRID, beta_grid=BETA_GRID):
   
    return pd.DataFrame(
        [[recompute_F(sp, sn, xi, n_nodes, a, b) for b in beta_grid]
         for a in alpha_grid],
        index=pd.Index(alpha_grid, name="alpha"),
        columns=pd.Index(beta_grid, name="beta"))
