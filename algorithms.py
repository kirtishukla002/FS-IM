def compute_objective(graph, probabilities, seed_set,
                      z, a, b, thresh, c_pos, r_neg, edge_sign,
                      communities, num_simulations=1,
                      alpha_param=0.7, beta_param=0.3,
                      b_pp=1.0, b_nn=0.5,
                      b_pn=-0.5, b_np=-0.5):

    total_pos = 0.0
    total_neg = 0.0
    total_xi  = 0.0

    V        = len(graph)
    num_comm = len(set(communities.values()))

    seed_set = set(seed_set)

    for _ in range(num_simulations):

        
        opinions, sp_run, sn_run = signed_fuzzy_ic(
            graph, probabilities, seed_set,
            z, a, b, thresh, c_pos, r_neg, edge_sign,
            b_pp, b_nn, b_pn, b_np
        )

        total_pos += sp_run
        total_neg += sn_run

       
        comm_pos_count = defaultdict(float)
        comm_size      = defaultdict(int)

        for v in graph:
            c = communities[v]
            comm_size[c] += 1

            if v not in seed_set and opinions[v] > 0.0:
                comm_pos_count[c] += 1.0

        xi_run = sum(
            comm_pos_count[c] / comm_size[c]
            for c in comm_size
            if comm_size[c] > 0
        ) / num_comm

        total_xi += xi_run

   
    n  = float(num_simulations)
    sp = total_pos / n
    sn = total_neg / n
    xi = total_xi  / n

  
    f = (
        alpha_param       * (sp / V)
        + (1 - alpha_param) * xi
        - beta_param       * (sn / V)
    )

    return f, sp, sn, xi

def dsim(graph, probabilities, seed_budget,
         z, a, b, thresh, c_pos, r_neg, edge_sign,
         communities, alpha=0.7, beta=0.3,
         simulations=10, random_seed=42,
         b_pp=1.0, b_nn=0.5,
         b_pn=-0.5, b_np=-0.5)
    random.seed(random_seed)

    start_time = time.time()

    seed_set = []
    selected = set()
    blocked = set()

    heap = []
    round_id = 0

    def average_spread(candidate_seed_set):

        total_F = 0.0

        for _ in range(simulations):

            f, _, _, _ = compute_objective(
                graph, probabilities, candidate_seed_set,
                z, a, b, thresh, c_pos, r_neg, edge_sign,
                communities, num_simulations=1,
                alpha_param=alpha, beta_param=beta,
                b_pp=b_pp, b_nn=b_nn,
                b_pn=b_pn, b_np=b_np
            )

            total_F += f

        return total_F / simulations

    
    for node in graph.nodes():

        gain = average_spread([node])

        heapq.heappush(
            heap,
            (-gain, round_id, node, 0)
        )

    
    for current_round in range(seed_budget):

        round_id += 1

        while heap:

            neg_gain, entry_round, node, flag = heapq.heappop(heap)
            gain = -neg_gain

           
            if node in selected:
                continue

            
            if node in blocked:
                continue

          
            if flag == current_round:
                seed_set.append(node)
                selected.add(node)

             
                opinions, _, _ = signed_fuzzy_ic(
                    graph, probabilities, [node],
                    z, a, b, thresh, c_pos, r_neg, edge_sign,
                    b_pp, b_nn, b_pn, b_np
                )

                for reached_node, level in opinions.items():
                    if level != 0:
                        blocked.add(reached_node)

                break

          
            new_gain = average_spread(seed_set + [node])

            heapq.heappush(
                heap,
                (-new_gain, round_id, node, current_round)
            )

    runtime = time.time() - start_time

    return seed_set, runtime
def greedy_im(G, probabilities, k,
              z, a, b, thresh, c_pos, r_neg, edge_sign,
              communities, num_sim=1, alpha=0.7, beta=0.3, verbose=True):

    S, log = set(), []
    current_value = 0.0

    total_communities = len({cid for cid in communities.values()})
    covered_comms = set()

    if verbose:
        print("\n" + "=" * 60)
        print("  GREEDY SEED SELECTION")
        print("=" * 60)

    for i in range(k):
        best_node, best_gain, best_stats = None, -float("inf"), None

        for v in G.nodes():
            if v in S:
                continue
            f_val, pos, neg, _ = compute_objective(
                G, probabilities, S | {v},
                z, a, b, thresh, c_pos, r_neg, edge_sign,
                communities,
                num_sim=num_sim, alpha=alpha, beta=beta
            )
            gain = f_val - current_value
            if gain > best_gain:
                best_node, best_gain, best_stats = v, gain, (f_val, pos, neg)

        if best_node is None:
            if verbose:
                print("  No more nodes to select.")
            break

        S.add(best_node)
        current_value = best_stats[0]
        node_comm = communities.get(best_node, -1)
        log.append((best_node, node_comm, best_gain, best_stats[1], best_stats[2]))

        if node_comm != -1:
            covered_comms.add(node_comm)

        if verbose:
            cov_pct = 100.0 * len(covered_comms) / total_communities if total_communities else 0.0
            print(f"  Greedy  [{i+1}/{k}]  node={best_node!s:>6}  "
                  f"comm={node_comm:>3}  gain={best_gain:.4f}  "
                  f"σ+={best_stats[1]:.4f}  σ-={best_stats[2]:.4f}  "
                  f"cov={cov_pct:.1f}%")

    return S, log

def celf_im(G, probabilities, k, z, a, b, thresh, c_pos, r_neg, edge_sign,
            communities, num_sim=1, alpha=0.7, beta=0.3):
    S, log = set(), []
    Q, cnt = [], itertools.count()          # counter = safe tie-breaker

    def evaluate(seed_set):
        return compute_objective_stats(
            G, probabilities, seed_set, z, a, b, thresh, c_pos, r_neg, edge_sign,
            communities, num_simulations=num_sim, alpha=alpha, beta=beta)

    for u in G.nodes():
        res = evaluate({u})
        heapq.heappush(Q, (-res["F_mean"], next(cnt), u, 0, res["sp_mean"], res["sn_mean"]))

    F_S = 0.0
    while len(S) < k and Q:
        neg_gain, _, u, last_iter, sp, sn = heapq.heappop(Q)
        if u in S:
            continue
        if last_iter == len(S):
            S.add(u)
            F_S += -neg_gain
            log.append((u, communities.get(u, -1), -neg_gain, sp, sn))
        else:
            res = evaluate(S | {u})
            dF = res["F_mean"] - F_S
            heapq.heappush(Q, (-dF, next(cnt), u, len(S), res["sp_mean"], res["sn_mean"]))
    return S, log


def community_aware_im(G, probabilities, k, communities, z, a, b, thresh, c_pos, r_neg,
                       edge_sign, num_sim=1, alpha=0.7, beta=0.3):
    pos_out = {n: sum(1 for _, v in G.out_edges(n) if edge_sign.get((n, v), 0) == 1)
               for n in G.nodes()}

    buckets = defaultdict(list)
    for node, cid in communities.items():
        if node in G:                                   # guard against stray nodes
            buckets[cid].append(node)
    for cid in buckets:
        buckets[cid].sort(key=lambda n: pos_out[n], reverse=True)

    order = sorted(buckets, key=lambda c: len(buckets[c]), reverse=True)
    ptr = dict.fromkeys(order, 0)
    seeds = set()

    while len(seeds) < k:                               # until k or exhausted
        progressed = False
        for cid in order:
            if len(seeds) >= k:
                break
            if ptr[cid] < len(buckets[cid]):
                seeds.add(buckets[cid][ptr[cid]])
                ptr[cid] += 1
                progressed = True
        if not progressed:
            break

    if len(seeds) < k:                                  # nodes without a community
        rem = sorted((n for n in G.nodes() if n not in seeds),
                     key=G.out_degree, reverse=True)
        seeds.update(rem[:k - len(seeds)])

    res = compute_objective_stats(G, probabilities, seeds, z, a, b, thresh, c_pos, r_neg,
                                  edge_sign, communities, num_simulations=num_sim,
                                  alpha=alpha, beta=beta)
    return seeds, [(s, communities.get(s, -1), 0.0, res["sp_mean"], res["sn_mean"])
                   for s in seeds]


def fuzzy_im(G, probabilities, k, z, thresh, c_pos, communities, num_sim=1):
    S, log = set(), []
    H, cnt = [], itertools.count()

    def spread(seed_set):
        return np.mean([sum(fuzzy_ic_propagation(G, probabilities, seed_set, z, thresh, c_pos).values())
                        for _ in range(num_sim)])

    for u in G.nodes():
        heapq.heappush(H, (-spread({u}), next(cnt), u, 0))   # set, not list

    cur_spread = 0.0
    for _ in range(k):
        best_node, best_gain = None, -float("inf")
        while H:
            neg_gain, _, u, flag = heapq.heappop(H)
            if flag == len(S):
                best_node, best_gain = u, -neg_gain
                break
            gain = spread(S | {u}) - cur_spread
            heapq.heappush(H, (-gain, next(cnt), u, len(S)))
        if best_node is None:
            break
        S.add(best_node)
        cur_spread += best_gain
        log.append((best_node, communities.get(best_node, -1), best_gain, cur_spread, 0.0))
    return S, log       
def degree_discount_im(G, k, probabilities, edge_sign,
                       communities, num_sim=1, alpha=0.7, beta=0.3,
                       z=None, a=None, b=None, t_param=None,
                       c_pos=None, r_neg=None, verbose=True):

    required = dict(z=z, a=a, b=b, t_param=t_param, c_pos=c_pos, r_neg=r_neg)
    missing = [n for n, v in required.items() if v is None]
    if missing:
        raise ValueError(f"degree_discount_im needs: {', '.join(missing)}")

    pos_out = defaultdict(list)
    for u, v in G.edges():
        if edge_sign[(u, v)] == 1:
            pos_out[u].append(v)

    d_pos = {u: len(pos_out[u]) for u in G.nodes()}
    p_bar = {u: (sum(probabilities[(u, v)] for v in pos_out[u]) / d_pos[u]) if d_pos[u] else 0.0
             for u in G.nodes()}
    t_count = defaultdict(int)

    def discount(u):
        du, tu, pu = d_pos[u], t_count[u], p_bar[u]
        return du - 2 * tu - (du - tu) * tu * pu

    S, log = set(), []

    if verbose:
        print("\n" + "=" * 60)
        print("  DEGREE DISCOUNT SEED SELECTION")
        print("=" * 60)

    for it in range(k):
        best_u, best_dd = None, -np.inf
        for u in G.nodes():
            if u in S:
                continue
            dd = discount(u)
            if dd > best_dd:
                best_u, best_dd = u, dd

        if best_u is None:
            break

        S.add(best_u)

        # nodes w with a positive edge w -> best_u lose one useful out-edge
        for w in G.predecessors(best_u):
            if edge_sign[(w, best_u)] == 1:
                t_count[w] += 1

        _, sp, sn, _ = compute_objective(
            G, probabilities, S, z, a, b, t_param, c_pos, r_neg, edge_sign,
            communities, num_sim=num_sim, alpha=alpha, beta=beta)
        comm = communities.get(best_u, -1)
        log.append((best_u, comm, best_dd, sp, sn))

        if verbose:
            print(f"  DD      [{it+1}/{k}]  node={best_u!s:>6}  comm={comm:>3}  "
                  f"dd={best_dd:.4f}  σ+={sp:.4f}  σ-={sn:.4f}")

    return S, log

def signed_pagerank_im(G, k, probabilities, edge_sign,
                       c_pos, r_neg, communities,
                       z, a, b, thresh,
                       num_sim=1, alpha=0.7, beta=0.3,
                       damping=0.85, reverse=True,
                       return_scores=True, verbose=True):

    Gw = nx.DiGraph()
    Gw.add_nodes_from(G.nodes())            
    for u, v in G.edges():
        w = (c_pos[(u, v)] if edge_sign[(u, v)] == 1
             else max(0.0, 1.0 - r_neg[(u, v)]))
        if reverse:
            Gw.add_edge(v, u, weight=w)      
        else:
            Gw.add_edge(u, v, weight=w)

    pr_scores = nx.pagerank(Gw, alpha=damping, weight="weight", max_iter=500)
    ranked = sorted(pr_scores.items(), key=lambda x: x[1], reverse=True)
    top_k = [node for node, _ in ranked[:k]]

    S, log = set(), []

    if verbose:
        print("\n" + "=" * 60)
        print("  SIGNED PAGERANK SEED SELECTION")
        print("=" * 60)

    for it, u in enumerate(top_k):
        S.add(u)
        _, sp, sn, _ = compute_objective(
            G, probabilities, S, z, a, b, thresh, c_pos, r_neg, edge_sign,
            communities, num_sim=num_sim, alpha=alpha, beta=beta)
        comm = communities.get(u, -1)
        log.append((u, comm, pr_scores[u], sp, sn))

        if verbose:
            print(f"  PR      [{it+1}/{k}]  node={u!s:>6}  comm={comm:>3}  "
                  f"PR={pr_scores[u]:.6f}  σ+={sp:.4f}  σ-={sn:.4f}")

    return (S, log, pr_scores) if return_scores else (S, log)

def signed_degree_im(G, k, probabilities, edge_sign,
                     communities, beta=0.3,
                     z=None, a=None, b=None, t_param=None,
                     c_pos=None, r_neg=None,
                     num_sim=1, alpha=0.7, verbose=True):

    required = dict(z=z, a=a, b=b, t_param=t_param, c_pos=c_pos, r_neg=r_neg)
    missing = [n for n, v in required.items() if v is None]
    if missing:
        raise ValueError(f"signed_degree_im needs: {', '.join(missing)}")

    d_pos, d_neg = defaultdict(int), defaultdict(int)
    for u, v in G.edges():
        if edge_sign[(u, v)] == 1:
            d_pos[u] += 1
        else:
            d_neg[u] += 1

    scores = {u: d_pos[u] - beta * d_neg[u] for u in G.nodes()}
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    top_k = [node for node, _ in ranked[:k]]

    S, log = set(), []

    if verbose:
        print("\n" + "=" * 60)
        print("  SIGNED DEGREE SEED SELECTION")
        print("=" * 60)

    for it, u in enumerate(top_k):
        S.add(u)
        _, sp, sn, _ = compute_objective(
            G, probabilities, S, z, a, b, t_param, c_pos, r_neg, edge_sign,
            communities, num_sim=num_sim, alpha=alpha, beta=beta)
        comm = communities.get(u, -1)
        log.append((u, comm, scores[u], sp, sn))

        if verbose:
            print(f"  SD      [{it+1}/{k}]  node={u!s:>6}  comm={comm:>3}  "
                  f"score={scores[u]:.4f}  σ+={sp:.4f}  σ-={sn:.4f}")

    return S, log
