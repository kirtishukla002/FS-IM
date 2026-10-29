import random
def balance_factor(u, v, edge_sign, opinions,
                   b_pp=1.0, b_nn=0.5,
                   b_pn=-0.5, b_np=-0.5):
    """
    Computes the balance factor based on the activation sign of u
    and the sign of edge (u, v).
    """
    s_uv = edge_sign[(u, v)]
    l_u = opinions[u]

    if l_u >= 0 and s_uv == 1:
        return b_pp        # positive source, positive edge

    elif l_u < 0 and s_uv == -1:
        return b_nn        # negative source, negative edge

    elif l_u >= 0 and s_uv == -1:
        return b_pn        # positive source, negative edge

    else:
        return b_np        # negative source, positive edge


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
