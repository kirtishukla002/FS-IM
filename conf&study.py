SEED = 42
SELECT_SEED_BASE = 10_000      
EVAL_SEED_BASE = 500_000       

SELECT_SIMS = 5              
POOL_SIZE = 100               
MC_EVAL_RUNS = 10             
BUDGETS = [5, 10, 15]
CA_CAND_PER_COMM = 10          
DD_MODE = "ic"                 
ABLATION_KS = [5, 10, 15]            
SENS_K = 10                    
RUN_ONLY = None                
RUN_SENSITIVITY = False        
def run_experiment_suite(dataset_name, G, probabilities, edge_sign, partition,
                         k_values=BUDGETS, mc_eval_runs=MC_EVAL_RUNS,
                         select_sims=SELECT_SIMS, pool_size=POOL_SIZE,
                         run_sensitivity=RUN_SENSITIVITY):
    print("\n" + "=" * 90)
    print(f"BENCHMARK SUITE: {dataset_name.upper()} (|V|={G.number_of_nodes():,}, |E|={G.number_of_edges():,})")
    print("=" * 90)

    params = initialize_parameters(G, probabilities)
    z, a, b, thresh, c_pos, r_neg = params
    pool = candidate_pool(G, pool_size)
    records, main_seeds = [], {}

    def evaluate(seeds, flags=None):
        return compute_objective_stats(G, probabilities, seeds, z, a, b, thresh, c_pos, r_neg, edge_sign,
                                       partition, num_simulations=mc_eval_runs,
                                       alpha=ALPHA_REF, beta=BETA_REF, ablation_flags=flags,
                                       sim_seed_base=EVAL_SEED_BASE)

    for k in k_values:
        print(f"\n>>> Running Evaluation for Budget k = {k} ...")
        algo_dict = {
            "DS-IM (Proposed)": lambda: ds_im(G, probabilities, k, *params[:3], thresh, c_pos, r_neg, edge_sign,
                                              partition, num_sim=select_sims, pool=pool),
            "Greedy-IM": lambda: greedy_im(G, probabilities, k, *params[:3], thresh, c_pos, r_neg, edge_sign,
                                           partition, num_sim=select_sims, pool=pool),
            "CELF": lambda: celf_im(G, probabilities, k, *params[:3], thresh, c_pos, r_neg, edge_sign,
                                    partition, num_sim=select_sims, pool=pool),
            "CA-Greedy": lambda: ca_greedy_im(G, probabilities, k, z, a, b, thresh, c_pos, r_neg, edge_sign,
                                              partition, num_sim=select_sims),
            "Fuzzy-IM": lambda: fuzzy_im(G, probabilities, k, z, thresh, c_pos, partition,
                                         num_sim=select_sims, pool=pool),
            "Signed-PageRank": lambda: signed_pagerank_im(G, k, edge_sign, c_pos, r_neg),
            "Signed-Degree": lambda: signed_degree_im(G, k, edge_sign),
            "Degree-Discount": lambda: degree_discount_im(G, k, edge_sign, probabilities, discount_mode=DD_MODE),
        }

        eval_res = {}
        for name, fn in algo_dict.items():
            t0 = time.time()
            seeds, _ = fn()
            runtime = time.time() - t0
            print(f"    [k={k}] {name:<18} selected in {runtime:8.1f}s", flush=True)
            if name.startswith("DS-IM"):
                main_seeds[k] = seeds
            st = evaluate(seeds)
            covered = {partition[s] for s in seeds if s in partition}
            st["coverage_pct"] = 100.0 * len(covered) / max(len(set(partition.values())), 1)
            st["runtime"] = runtime
            eval_res[name] = st

        base = eval_res["DS-IM (Proposed)"]["raw_F"]
        print(f"\n[Results: k = {k} | Dataset = {dataset_name}]")
        print(f"{'Algorithm':<18} {'F(S) (μ±σ)':<16} {'σ+ (μ±σ)':<18} {'σ- (μ±σ)':<16} "
              f"{'ξ':<8} {'Reach% (μ±σ)':<16} {'CReach%':<10} {'SeedCov%':<9} {'Time(s)':<9} {'p-val(F)':<10}")
        print("-" * 135)
        for name, m in eval_res.items():
            if name.startswith("DS-IM"):
                p_val, p_str = float("nan"), "Baseline"
            else:
                _, p_val = stats.ttest_rel(base, m["raw_F"])   # paired: shared CRN streams
                p_str = f"{p_val:.4e}"
            reach_str = f"{m['reach_mean']:.1f}±{m['reach_std']:.1f}"
            print(f"{name:<18} {m['F_mean']:.3f}±{m['F_std']:.3f}   "
                  f"{m['sp_mean']:.1f}±{m['sp_std']:.1f}   "
                  f"{m['sn_mean']:.1f}±{m['sn_std']:.1f}   "
                  f"{m['xi_mean']:.3f}  {reach_str:<16}{m['creach_mean']:<10.1f}"
                  f"{m['coverage_pct']:<9.1f}{m['runtime']:<9.2f}{p_str:<10}")
            records.append({"dataset": dataset_name, "k": k, "algo": name, **m, "p_val": p_val})

    ab_ks = list(k_values) if ABLATION_KS is None else [kk for kk in ABLATION_KS]
    variants = {
        "Full DS-IM": {},
        "No Community Term (alpha=1.0)": {"disable_diversity": True},
        "No Negative Suppression (beta=0.0)": {"disable_beta": True},
        "No Structural Balance": {"disable_balance": True},
        "Discrete Binary Threshold": {"binary_threshold": True},
        "Diverification":  {"Diversification": True},
    }
    ablation_results = {}
    for ab_k in ab_ks:
       
        print(f"\n--- Component Ablation ({dataset_name}, k = {ab_k}; all variants scored under the full model/objective) ---")
        for name, flags in variants.items():
            if not flags and ab_k in main_seeds:
                seeds = main_seeds[ab_k]                     
            else:
                seeds, _ = ds_im(G, probabilities, ab_k, z, a, b, thresh, c_pos, r_neg, edge_sign, partition,
                                 num_sim=select_sims, ablation_flags=flags, pool=pool)
            st = evaluate(seeds)                             
            ablation_results[(ab_k, name)] = st
            print(f"  {name:<38} | F = {st['F_mean']:.4f}±{st['F_std']:.3f} | σ+ = {st['sp_mean']:.1f} "
                  f"| σ- = {st['sn_mean']:.1f} | ξ = {st['xi_mean']:.3f}")

    if run_sensitivity:
        parameter_sensitivity(dataset_name, G, probabilities, edge_sign, partition, params, pool,
                              k=SENS_K, select_sims=select_sims, mc_eval_runs=mc_eval_runs)

    return records, ablation_results


def parameter_sensitivity(dataset_name, G, probabilities, edge_sign, partition, params, pool,
                          k=10, select_sims=SELECT_SIMS, mc_eval_runs=MC_EVAL_RUNS,
                          alpha_grid=(0.2, 0.5, 0.5, 0.7, 1.0), beta_grid=(0.1, 0.3, 0.5, 0.7),
                          out_dir="results_plots"):
    
    os.makedirs(out_dir, exist_ok=True)
    z, a, b, thresh, c_pos, r_neg = params
    cache, rows = {}, []

    def run(al, be):
        if (al, be) not in cache:
            seeds, _ = ds_im(G, probabilities, k, z, a, b, thresh, c_pos, r_neg, edge_sign, partition,
                             num_sim=select_sims, alpha=al, beta=be, pool=pool)
            cache[(al, be)] = compute_objective_stats(
                G, probabilities, seeds, z, a, b, thresh, c_pos, r_neg, edge_sign, partition,
                num_simulations=mc_eval_runs, alpha=ALPHA_REF, beta=BETA_REF, sim_seed_base=EVAL_SEED_BASE)
        return cache[(al, be)]

    print(f"\n--- Parameter Sensitivity ({dataset_name}, k = {k}) ---")
    print(f"{'param':<7}{'value':<8}{'F_ref':<14}{'σ+':<10}{'σ-':<10}{'ξ':<8}")
    for al in alpha_grid:
        rows.append(("alpha", al, run(al, BETA_REF)))
    for be in beta_grid:
        rows.append(("beta", be, run(ALPHA_REF, be)))
    for p, v, m in rows:
        print(f"{p:<7}{v:<8}{m['F_mean']:.4f}±{m['F_std']:.3f}  {m['sp_mean']:<10.1f}{m['sn_mean']:<10.1f}{m['xi_mean']:<8.3f}")

    with open(os.path.join(out_dir, f"{dataset_name}_sensitivity.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["param", "value", "F_ref_mean", "F_ref_std", "sp", "sn", "xi"])
        for p, v, m in rows:
            w.writerow([p, v, m["F_mean"], m["F_std"], m["sp_mean"], m["sn_mean"], m["xi_mean"]])

    fig, axes = plt.subplots(2, 4, figsize=(15, 6))
    for r, (pname, grid) in enumerate((("alpha", alpha_grid), ("beta", beta_grid))):
        sel = [m for p, v, m in rows if p == pname]
        for c, (key, lab) in enumerate((("F", r"$F_{ref}$"), ("sp", r"$\sigma^+$"),
                                        ("sn", r"$\sigma^-$"), ("xi", r"$\xi$"))):
            axes[r, c].errorbar(grid, [m[f"{key}_mean"] for m in sel], yerr=[m[f"{key}_std"] for m in sel],
                                marker="o", capsize=3, lw=1.5, color="#d62728")
            axes[r, c].set_xlabel(r"$\alpha$" if pname == "alpha" else r"$\beta$")
            axes[r, c].set_ylabel(lab)
            axes[r, c].grid(True, linestyle="--", alpha=0.5)
    plt.suptitle(f"Parameter sensitivity of DS-IM ({dataset_name}, k={k})")
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, f"{dataset_name}_sensitivity.png"), dpi=300)
    fig.savefig(os.path.join(out_dir, f"{dataset_name}_sensitivity.pdf"), dpi=300)
    plt.close(fig)
