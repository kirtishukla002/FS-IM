import random
def balance_factor(u, v, edge_sign, opinions,
                   b_pp=1.0, b_nn=0.5,
                   b_pn=-0.5, b_np=-0.5):
    
    s_uv = edge_sign[(u, v)]
    l_u = opinions[u]

    if l_u >= 0 and s_uv == 1:
        return b_pp       

    elif l_u < 0 and s_uv == -1:
        return b_nn       

    elif l_u >= 0 and s_uv == -1:
        return b_pn        

    else:
        return b_np       


def signed_fuzzy_ic(graph, probabilities, seed_set,
                    z, a, b, thresh, c_pos, r_neg, edge_sign,
                    b_pp=1.0, b_nn=0.5,
                    b_pn=-0.5, b_np=-0.5):

    opinions   = {node: 0.0 for node in graph}
    active_set = set(seed_set)

    for s in seed_set:
        opinions[s] = z[s]

    rounds = 0
    max_rounds = graph.number_of_nodes()

    while active_set and rounds < max_rounds:

        rounds += 1
        current_active = list(active_set)

        for u in current_active:
            for v in graph[u]:
                if v in active_set:
                    continue

                if random.random() > probabilities[(u, v)]:
                    continue

                x   = random.uniform(0, abs(opinions[u]))
                a_uv = a[(u, v)]
                b_uv = b[(u, v)]
                s_uv = edge_sign[(u, v)]
                w_uv = probabilities[(u, v)]

                if x <= a_uv:
                    Iuv = 0.0
                elif x <= b_uv:
                    Iuv = (c_pos[(u, v)] * w_uv if s_uv == 1
                           else r_neg[(u, v)] * w_uv)
                else:
                    Iuv = 1.0

                Iuv *= s_uv
                Iuv *= balance_factor(u, v, edge_sign, opinions,
                                      b_pp, b_nn, b_pn, b_np)

                tv = thresh[v]
                zv = z[v]

                if Iuv < -tv:
                    Lv = -1.0
                elif -tv < Iuv < 0:
                    Lv = -(zv * abs(Iuv)) / tv
                elif abs(Iuv) < 1e-12:
                    Lv = 0.0
                elif 0 < Iuv < tv:
                    Lv = (zv * Iuv) / tv
                else:
                    Lv = 1.0

                Lv = max(-1.0, min(1.0, Lv))

                if abs(Lv) > abs(opinions[v]):
                    opinions[v] = Lv
                    active_set.add(v)

            active_set.discard(u)

    seed_set = set(seed_set)

    positive_count = sum(
        1 for v in opinions
        if v not in seed_set and opinions[v] > 0
    )

    negative_count = sum(
        1 for v in opinions
        if v not in seed_set and opinions[v] < 0
    )

    return opinions, positive_count, negative_count
                      
def snic_propagation(G, probabilities, edge_sign, seed_set,
                     realization=None, rng=random):
    opinions = {node: 0 for node in G}
    visited = set(seed_set)
    for s in seed_set:
        opinions[s] = 1

    edge_rand = realization["edge_rand"] if realization else None
    prio = realization["prio"] if realization else None

    active = list(seed_set)
    while active:
        if prio is not None:
            active.sort(key=prio.__getitem__)       # fixed tie order
        else:
            rng.shuffle(active)
        nxt = []
        for u in active:
            u_sign = opinions[u]
            for v in G[u]:
                if v in visited:
                    continue
                r = edge_rand[(u, v)] if edge_rand is not None else rng.random()
                if r <= probabilities[(u, v)]:
                    s_uv = edge_sign.get((u, v), 1)
                    opinions[v] = 1 if (u_sign * s_uv) > 0 else -1
                    visited.add(v)
                    nxt.append(v)
        active = nxt
    return opinions


def slt_propagation(G, edge_sign, seed_set, in_weights,
                    thresholds=None, max_iters=None, rng=random):
    opinions = {node: 0 for node in G}
    active = set(seed_set)
    for s in seed_set:
        opinions[s] = 1

    if thresholds is None:
        thresholds = {v: rng.uniform(SLT_TH_LOW, SLT_TH_HIGH) for v in G.nodes()}

    pos_acc, neg_acc = defaultdict(float), defaultdict(float)
    frontier, rounds = list(seed_set), 0

    while frontier and (max_iters is None or rounds < max_iters):
        rounds += 1
        touched = set()
        for u in frontier:
            lu = opinions[u]
            for v in G[u]:
                if v in active:
                    continue
                w = in_weights[v][u]
                if lu * edge_sign.get((u, v), 1) > 0:
                    pos_acc[v] += w
                else:
                    neg_acc[v] += w
                touched.add(v)

        newly = {}
        for v in touched:
            diff, th = pos_acc[v] - neg_acc[v], thresholds[v]
            if diff >= th:
                newly[v] = 1
            elif -diff >= th:
                newly[v] = -1

        for v, sgn in newly.items():
            opinions[v] = sgn
            active.add(v)
        frontier = list(newly)
    return opinions


def simulate(model_name, G, seeds, P, realization=None, rng=random):
    if model_name == "fs_ic":
        return signed_fuzzy_ic(G, P["prob"], seeds, P["z"], P["a"], P["b"],
                               P["thresh"], P["c_pos"], P["r_neg"], P["edge_sign"])
    if model_name == "snic":
        return snic_propagation(G, P["prob"], P["edge_sign"], seeds,
                                realization=realization, rng=rng)
    if model_name == "s_lt":
        th = realization["thresholds"] if realization else None
        return slt_propagation(G, P["edge_sign"], seeds, P["in_weights"],
                               thresholds=th, rng=rng)
    raise ValueError(f"Unknown diffusion model: {model_name}")


def spread_counts(opinions, seeds):
    seeds = set(seeds)
    sp = sum(1 for v, o in opinions.items() if v not in seeds and o > 0)
    sn = sum(1 for v, o in opinions.items() if v not in seeds and o < 0)
    return sp, sn
