"""Exact finite-tree checks for ChunkRatio (ChunkRatio-PD) v1.1. Standard library only.

These are mathematical checks of the v1.1 identities, bounds and fixtures.
They are not LLM experiments and say nothing about benchmark performance.

Notation follows the v1.1 method document (dimensionless log-ratios):
  T+ aligned teacher, T0 its reference, rho frozen student reference.
  d_s(z)   = log T+(z|s) - log T0(z|s)          future teacher displacement
  l_s(z)   = log rho(z|s) - log T0(z|s)          future reference mismatch
  phi(s)   = log E_rho[e^d] = log E_T+[e^l]      exact residual
  mu_rho   = E_rho[d]   (v1.1 main correction)
  mu_T+    = E_T+[l]    (v1.0 direction, ablation)
  target   pi*(y) ∝ rho(y) T+(y)/T0(y)

Run:  python ChunkRatio_Math_Checks.py [output.json]
"""
import itertools
import json
import math
import random
import sys
from pathlib import Path

EOS = 2            # token 2 is EOS; tokens 0 and 1 are ordinary
VOCAB = (0, 1, EOS)
H = 5              # global response horizon (number of generated tokens, EOS included)
TOL = 1e-11


# ---------------------------------------------------------------- tree models
def terminal(s):
    return len(s) == H or (len(s) > 0 and s[-1] == EOS)


def nonterminal_prefixes():
    out, frontier = [()], [()]
    while frontier:
        nxt = []
        for s in frontier:
            for a in VOCAB:
                u = s + (a,)
                if not terminal(u):
                    out.append(u)
                    nxt.append(u)
        frontier = nxt
    return out


PREFIXES = nonterminal_prefixes()


def softmax(z):
    m = max(z)
    e = [math.exp(v - m) for v in z]
    t = math.fsum(e)
    return [v / t for v in e]


def make_world(rng, a, b, base_sd=0.6):
    """T0 from random logits; T+ = T0 logits + a*N; rho = T0 logits + b*N'."""
    T0, Tp, rho = {}, {}, {}
    for s in PREFIXES:
        base = [rng.gauss(0, base_sd) for _ in VOCAB]
        n1 = [rng.gauss(0, 1) for _ in VOCAB]
        n2 = [rng.gauss(0, 1) for _ in VOCAB]
        T0[s] = softmax(base)
        Tp[s] = softmax([x + a * y for x, y in zip(base, n1)])
        rho[s] = softmax([x + b * y for x, y in zip(base, n2)])
    return T0, Tp, rho


def logp(m, s, c):
    out, cur = 0.0, tuple(s)
    for tok in c:
        out += math.log(m[cur][tok])
        cur += (tok,)
    return out


def completions(s):
    """All suffixes z with s+z terminal (prefix-free, sums to one under any model)."""
    if terminal(s):
        return [()]
    out = []
    for a in VOCAB:
        for z in completions(s + (a,)):
            out.append((a,) + z)
    return out


def chunks(s, rule):
    """Complete prefix-free chunk set at state s for a causal stopping rule."""
    L = rule(s)
    out = []

    def grow(c):
        u = s + c
        if c and (terminal(u) or len(c) == L):
            out.append(c)
            return
        for a in VOCAB:
            grow(c + (a,))
    grow(())
    return out


def fixed(L):
    return lambda s: L


def causal_variable(s):
    return 1 if len(s) % 2 == 0 else 2      # depends only on the current prefix


RULES = {"L=1": fixed(1), "L=2": fixed(2), "L=3": fixed(3), "L=H": fixed(H), "causal": causal_variable}


# ---------------------------------------------------------------- quantities
class World:
    def __init__(self, T0, Tp, rho):
        self.T0, self.Tp, self.rho = T0, Tp, rho
        self._c = {}

    def d(self, s, z):
        return logp(self.Tp, s, z) - logp(self.T0, s, z)

    def ell(self, s, z):
        return logp(self.rho, s, z) - logp(self.T0, s, z)

    def stats(self, s):
        if s in self._c:
            return self._c[s]
        if terminal(s):
            r = dict(phi=0.0, phi_alt=0.0, mu_rho=0.0, var_rho=0.0, mu_tp=0.0, d_range=0.0)
        else:
            Z = completions(s)
            pr = [math.exp(logp(self.rho, s, z)) for z in Z]
            pt = [math.exp(logp(self.Tp, s, z)) for z in Z]
            dv = [self.d(s, z) for z in Z]
            lv = [self.ell(s, z) for z in Z]
            phi = math.log(math.fsum(p * math.exp(x) for p, x in zip(pr, dv)))
            phi_alt = math.log(math.fsum(p * math.exp(x) for p, x in zip(pt, lv)))
            mu = math.fsum(p * x for p, x in zip(pr, dv))
            var = math.fsum(p * (x - mu) ** 2 for p, x in zip(pr, dv))
            mu_tp = math.fsum(p * x for p, x in zip(pt, lv))
            r = dict(phi=phi, phi_alt=phi_alt, mu_rho=mu, var_rho=var, mu_tp=mu_tp,
                     d_range=max(dv) - min(dv))
        self._c[s] = r
        return r

    def phi(self, s):
        return self.stats(s)["phi"]

    def target(self):
        Y = completions(())
        w = [logp(self.rho, (), y) + self.d((), y) for y in Y]
        m = max(w)
        e = [math.exp(v - m) for v in w]
        t = math.fsum(e)
        return {y: v / t for y, v in zip(Y, e)}

    def chunk_policy(self, s, rule, head):
        """q(c|s) ∝ rho(c|s) exp(delta_T(c|s) + head(sc)); head is 0 at terminal endpoints."""
        cs = chunks(s, rule)
        sc = [logp(self.rho, s, c) + self.d(s, c) + (0.0 if terminal(s + c) else head(s + c)) for c in cs]
        m = max(sc)
        e = [math.exp(v - m) for v in sc]
        t = math.fsum(e)
        return {c: v / t for c, v in zip(cs, e)}


def walk(world, rule, head, visit=None):
    """Distribution over complete responses induced by a chunk policy; optional state visitor."""
    out = {}

    def rec(s, p):
        if terminal(s):
            out[s] = out.get(s, 0.0) + p
            return
        q = world.chunk_policy(s, rule, head)
        if visit:
            visit(s, p, q)
        for c, qc in q.items():
            rec(s + c, p * qc)
    rec((), 1.0)
    return out


def kl(p, q):
    return math.fsum(pv * math.log(pv / q[y]) for y, pv in p.items() if pv > 0)


def jensen_gap(probs, f):
    """J_p(f) = log E_p e^f - E_p f for dicts keyed alike."""
    m = max(f.values())
    lse = m + math.log(math.fsum(probs[k] * math.exp(f[k] - m) for k in probs))
    return lse - math.fsum(probs[k] * f[k] for k in probs)


def objective(world, pi):
    """J(pi) = E_pi d_T(y) - KL(pi || rho)."""
    return math.fsum(p * (world.d((), y) - (math.log(p) - logp(world.rho, (), y)))
                     for y, p in pi.items() if p > 0)


# ---------------------------------------------------------------- checks
def check_identities(results):
    rng = random.Random(2600)
    errs = dict(two_representations=0.0, chunk_decomposition=0.0, partition_invariance=0.0,
                objective_identity=0.0, q_identity=0.0, forward_jensen=0.0, reverse_jensen=0.0,
                tilted_variance=0.0)
    bound_ratio = dict(hoeffding_forward=0.0, hoeffding_reverse=0.0, gap_upper=0.0,
                       fourth_order_local=0.0, sibling_regret=0.0)
    gap_min = math.inf
    for seed in range(4):
        world = World(*make_world(rng, 0.6, 0.6))
        pistar = world.target()
        # (4) two representations of phi
        for s in PREFIXES:
            st = world.stats(s)
            errs["two_representations"] = max(errs["two_representations"], abs(st["phi"] - st["phi_alt"]))
        # (5) exact chunk decomposition and partition invariance
        for name, rule in RULES.items():
            q = walk(world, rule, world.phi)
            errs["partition_invariance"] = max(errs["partition_invariance"],
                                               max(abs(q.get(y, 0) - pistar[y]) for y in pistar))
            for s in PREFIXES:
                mass_s = math.fsum(p for y, p in pistar.items() if y[:len(s)] == s)
                for c in chunks(s, rule):
                    mass_sc = math.fsum(p for y, p in pistar.items() if y[:len(s) + len(c)] == s + c)
                    lhs = math.log(mass_sc / mass_s) - logp(world.rho, s, c)
                    rhs = world.d(s, c) + world.phi(s + c) - world.phi(s)
                    errs["chunk_decomposition"] = max(errs["chunk_decomposition"], abs(lhs - rhs))
        # (11) J(pi*) - J(pi) = KL(pi || pi*) for random policies
        for _ in range(5):
            w = {y: rng.random() + 1e-3 for y in pistar}
            t = math.fsum(w.values())
            pi = {y: v / t for y, v in w.items()}
            errs["objective_identity"] = max(errs["objective_identity"],
                                             abs(objective(world, pistar) - objective(world, pi) - kl(pi, pistar)))
        # Section 8: Q^rho(s,c) = D_T(s) + delta_T(c|s) + mu_rho(sc)
        for s in PREFIXES:
            for c in chunks(s, fixed(2)):
                u = s + c
                Qr = math.fsum(math.exp(logp(world.rho, u, z)) * world.d((), u + z) for z in completions(u))
                rhs = world.d((), s) + world.d(s, c) + world.stats(u)["mu_rho"]
                errs["q_identity"] = max(errs["q_identity"], abs(Qr - rhs))
        # (12) Jensen-gap identities and bounds for several heads
        noise = {s: rng.gauss(0, 0.4) for s in PREFIXES}
        heads = {"zero": lambda u: 0.0,
                 "mu_rho": lambda u: world.stats(u)["mu_rho"],
                 "mu_tp": lambda u: world.stats(u)["mu_tp"],
                 "mu_rho_noisy": lambda u: world.stats(u)["mu_rho"] + noise[u]}
        for hname, head in heads.items():
            for rname, rule in RULES.items():
                e = lambda u: (0.0 if terminal(u) else head(u)) - world.phi(u)
                q = walk(world, rule, head)
                acc = dict(fwd=0.0, rev=0.0, hf=0.0, hr=0.0, reg_bound=0.0)

                def vis_fwd(s, p, qs):
                    exact = world.chunk_policy(s, rule, world.phi)
                    f = {c: e(s + c) for c in exact}
                    acc["fwd"] += p * jensen_gap(exact, f)
                    acc["hf"] += p * (max(f.values()) - min(f.values())) ** 2 / 8

                def vis_rev(s, p, qs):
                    f = {c: -e(s + c) for c in qs}
                    acc["rev"] += p * jensen_gap(qs, f)
                    acc["hr"] += p * (max(f.values()) - min(f.values())) ** 2 / 8
                    if hname in ("mu_rho", "mu_rho_noisy"):
                        g = {c: world.stats(s + c)["phi"] - world.stats(s + c)["mu_rho"] for c in qs}
                        omega = max(g.values()) - min(g.values())
                        eta = max(abs(noise[s + c]) if (hname == "mu_rho_noisy" and not terminal(s + c)) else 0.0
                                  for c in qs)
                        acc["reg_bound"] += p * (omega + 2 * eta) ** 2 / 8
                walk(world, rule, world.phi, vis_fwd)
                walk(world, rule, head, vis_rev)
                kf, kr = kl(pistar, q), kl(q, pistar)
                errs["forward_jensen"] = max(errs["forward_jensen"], abs(kf - acc["fwd"]))
                errs["reverse_jensen"] = max(errs["reverse_jensen"], abs(kr - acc["rev"]))
                if acc["hf"] > 0:
                    bound_ratio["hoeffding_forward"] = max(bound_ratio["hoeffding_forward"], kf / acc["hf"])
                if acc["hr"] > 0:
                    bound_ratio["hoeffding_reverse"] = max(bound_ratio["hoeffding_reverse"], kr / acc["hr"])
                if acc["reg_bound"] > 0:   # (15a): regret = KL(q||pi*) <= 1/8 E_q sum (omega + 2 eta)^2
                    bound_ratio["sibling_regret"] = max(bound_ratio["sibling_regret"], kr / acc["reg_bound"])
        # (13), (14): 0 <= phi - mu_rho <= eps^2/8 and local KL <= eps^4/512
        for s in PREFIXES:
            st = world.stats(s)
            g = st["phi"] - st["mu_rho"]
            gap_min = min(gap_min, g)
            if st["d_range"] > 0:
                bound_ratio["gap_upper"] = max(bound_ratio["gap_upper"], g / (st["d_range"] ** 2 / 8))
        for rule in RULES.values():
            for s in PREFIXES:
                cs = chunks(s, rule)
                eps = max(world.stats(s + c)["d_range"] for c in cs)
                if eps == 0:
                    continue
                exact = world.chunk_policy(s, rule, world.phi)
                qmu = world.chunk_policy(s, rule, lambda u: world.stats(u)["mu_rho"])
                local = max(kl(exact, qmu), kl(qmu, exact))
                bound_ratio["fourth_order_local"] = max(bound_ratio["fourth_order_local"], local / (eps ** 4 / 512))
        # (15b) tilted-variance identity, Simpson rule with 2000 intervals
        for s in PREFIXES[:12]:
            Z = completions(s)
            pr = [math.exp(logp(world.rho, s, z)) for z in Z]
            dv = [world.d(s, z) for z in Z]

            def var_t(t):
                w = [p * math.exp(t * x) for p, x in zip(pr, dv)]
                tot = math.fsum(w)
                m1 = math.fsum(wi * x for wi, x in zip(w, dv)) / tot
                return math.fsum(wi * (x - m1) ** 2 for wi, x in zip(w, dv)) / tot
            n = 2000
            hstep = 1.0 / n
            integ = var_t(0) * 1 + var_t(1) * 0
            for k in range(1, n):
                t = k * hstep
                integ += (4 if k % 2 else 2) * (1 - t) * var_t(t)
            integ *= hstep / 3
            st = world.stats(s)
            errs["tilted_variance"] = max(errs["tilted_variance"], abs((st["phi"] - st["mu_rho"]) - integ))
    results["identity_max_errors"] = errs
    results["bound_max_ratios_must_be_le_1"] = bound_ratio
    results["min_gap_phi_minus_mu_rho_must_be_ge_0"] = gap_min
    for k, v in errs.items():
        assert v < (1e-9 if k == "tilted_variance" else TOL), (k, v)
    for k, v in bound_ratio.items():
        assert v <= 1 + 1e-9, (k, v)
    assert gap_min >= -TOL


def check_routing(results):
    rng = random.Random(7)
    T0, Tp, rho = make_world(rng, 0.8, 0.8)
    # Route 1: T+ = T0 -> phi = mu_rho = 0, pi* = rho
    w1 = World(T0, dict(T0), rho)
    r1 = max(max(abs(w1.stats(s)["phi"]), abs(w1.stats(s)["mu_rho"])) for s in PREFIXES)
    pist = w1.target()
    r1b = max(abs(p - math.exp(logp(rho, (), y))) for y, p in pist.items())
    # Route 2: rho = T0 -> phi = 0 but mu_rho = -KL(T0||T+) on suffixes, generally nonzero
    w2 = World(T0, Tp, dict(T0))
    phi_max = max(abs(w2.stats(s)["phi"]) for s in PREFIXES)
    kl_err, mu_abs = 0.0, 0.0
    for s in PREFIXES:
        Z = completions(s)
        klv = math.fsum(math.exp(logp(T0, s, z)) * (logp(T0, s, z) - logp(Tp, s, z)) for z in Z)
        kl_err = max(kl_err, abs(w2.stats(s)["mu_rho"] + klv))
        mu_abs = max(mu_abs, abs(w2.stats(s)["mu_rho"]))
    results["routing"] = dict(no_alignment_update_max_phi_mu=r1, no_alignment_update_target_equals_rho=r1b,
                              shared_reference_max_phi=phi_max, shared_reference_mu_equals_minus_kl_err=kl_err,
                              shared_reference_max_abs_mu_rho=mu_abs)
    assert r1 < TOL and r1b < TOL and phi_max < TOL and kl_err < TOL and mu_abs > 1e-3


def check_regimes(results):
    """Reverse KL(q||pi*) with one-token chunks; identity shortcuts deliberately disabled."""
    regimes = {"small_displacement_large_mismatch": (0.08, 1.0),
               "large_displacement_small_mismatch": (1.0, 0.08),
               "both_moderate": (0.5, 0.5),
               "zero_displacement": (0.0, 1.0),
               "shared_reference": (1.0, 0.0)}
    table = {}
    for name, (a, b) in regimes.items():
        rng = random.Random(1000 + int(100 * a) + int(10 * b))
        acc = {"raw": [], "mu_tp_v1.0": [], "mu_rho_v1.1": [], "mu_rho_plus_half_var": []}
        for _ in range(6):
            world = World(*make_world(rng, a, b))
            pistar = world.target()
            heads = {"raw": lambda u: 0.0,
                     "mu_tp_v1.0": lambda u: world.stats(u)["mu_tp"],
                     "mu_rho_v1.1": lambda u: world.stats(u)["mu_rho"],
                     "mu_rho_plus_half_var": lambda u: world.stats(u)["mu_rho"] + 0.5 * world.stats(u)["var_rho"]}
            for k, hd in heads.items():
                acc[k].append(kl(walk(world, fixed(1), hd), pistar))
        table[name] = {k: math.fsum(v) / len(v) for k, v in acc.items()}
    results["regime_mean_reverse_kl_L1"] = table
    t = table
    assert t["zero_displacement"]["raw"] < TOL and t["zero_displacement"]["mu_rho_v1.1"] < TOL
    assert t["shared_reference"]["raw"] < TOL and t["shared_reference"]["mu_tp_v1.0"] < TOL
    r = t["small_displacement_large_mismatch"]
    assert r["mu_rho_v1.1"] < r["raw"] < r["mu_tp_v1.0"]
    r = t["large_displacement_small_mismatch"]
    assert r["mu_tp_v1.0"] < r["mu_rho_v1.1"]


def check_loss(results):
    def sigmoid(x):
        return 1 / (1 + math.exp(-x))

    def softplus(x):
        return max(x, 0) + math.log1p(math.exp(-abs(x)))

    def h(r):
        return r * math.log(r) - (1 + r) * math.log1p(r)

    def hp(r):
        return math.log(r) - math.log1p(r)

    def pair_loss(m, p):
        return softplus(m) - p * m

    # Bregman (logistic generator) equals soft BCE minus target entropy
    err = 0.0
    for p in (.1, .3, .5, .8, .95):
        Rt = (1 - p) / p
        ent = -p * math.log(p) - (1 - p) * math.log1p(-p)
        for m in (-4., -1., 0., 1., 4.):
            R = math.exp(-m)
            breg = h(Rt) - h(R) - hp(R) * (Rt - R)
            err = max(err, abs(p * breg - (pair_loss(m, p) - ent)))
    # numeric fixture from the coding spec
    alpha = 0.3
    tc, td = 0.8 + 0.1, 0.2 - 0.1
    logit = alpha * (tc - td)
    p = sigmoid(logit)
    loss0 = pair_loss(0.0, p)
    grad0 = sigmoid(0.0) - p
    # swap symmetry and common-offset invariance
    rng = random.Random(3)
    sym = off = 0.0
    for _ in range(1000):
        ac, ad = rng.uniform(-3, 3), rng.uniform(-3, 3)
        sc, sd, k = rng.uniform(-3, 3), rng.uniform(-3, 3), rng.uniform(-5, 5)
        m, pt = alpha * (ac - ad), sigmoid(alpha * (sc - sd))
        ms, ps = alpha * (ad - ac), sigmoid(alpha * (sd - sc))
        sym = max(sym, abs(pair_loss(m, pt) - pair_loss(ms, ps)))
        po = sigmoid(alpha * ((sc + k) - (sd + k)))
        off = max(off, abs(pair_loss(m, pt) - pair_loss(m, po)))
    results["loss"] = dict(bregman_bce_max_error=err, fixture_target_logit=logit, fixture_target_p=p,
                           fixture_loss_at_init=loss0, fixture_grad_at_init=grad0,
                           swap_symmetry_max_error=sym, common_offset_max_error=off)
    assert err < 1e-12 and abs(logit - 0.24) < 1e-12 and abs(p - 0.55971365) < 1e-8
    assert abs(loss0 - math.log(2)) < 1e-12 and abs(grad0 + 0.05971365) < 1e-8
    assert sym < 1e-12 and off < 1e-12

    # label error: endpoint errors <= eps give |dp| <= alpha*eps/2
    eps = 0.07
    mx = 0.0
    for _ in range(10000):
        u, v = rng.uniform(0, 1.5), rng.uniform(0, 1.5)
        du, dv = rng.uniform(-eps, eps), rng.uniform(-eps, eps)
        for a in (0.3, 1.0):
            diff = abs(sigmoid(a * (u + du - v - dv)) - sigmoid(a * (u - v)))
            assert diff <= a * eps / 2 + 1e-12
            mx = max(mx, diff)
    results["label_error"] = dict(observed_max=mx, bound_alpha_1=eps / 2)


def check_data_contracts(results):
    # suffix returns fixture: [0.3, -0.2, 0.4] -> [0.5, 0.2, 0.4], terminal 0
    inc = [0.3, -0.2, 0.4]
    ret = [math.fsum(inc[t:]) for t in range(len(inc))] + [0.0]
    assert all(abs(a - b) < 1e-12 for a, b in zip(ret, [0.5, 0.2, 0.4, 0.0]))
    # hierarchical weights sum to one
    rng = random.Random(5)
    total = 0.0
    prompts = [[rng.randint(2, 4) for _ in range(rng.randint(1, 6))] for _ in range(7)]
    for states in prompts:
        for k in states:
            total += (1 / len(prompts)) * (1 / len(states)) * len(list(itertools.combinations(range(k), 2))) / math.comb(k, 2)
    assert abs(total - 1) < 1e-12
    # products of local binary probabilities are not complementary
    fwd, rev = .8 * .8, .2 * .2
    # pair odds on a subset do not identify the subset's mass
    q1, q2 = [.3, .3, .4], [.1, .1, .8]
    assert math.isclose(q1[0] / q1[1], q2[0] / q2[1])
    results["data_contracts"] = dict(suffix_returns=ret, hierarchical_weight_total=total,
                                     product_probability=dict(forward=fwd, reverse=rev, total=fwd + rev),
                                     candidate_mass=dict(first=q1[0] + q1[1], second=q2[0] + q2[1]))


def main():
    results = {"method_version": "1.1", "tree": dict(vocab=list(VOCAB), eos=EOS, horizon=H,
                                                   n_nonterminal_states=len(PREFIXES),
                                                   n_complete_responses=len(completions(())))}
    check_identities(results)
    check_routing(results)
    check_regimes(results)
    check_loss(results)
    check_data_contracts(results)
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).with_suffix(".json")
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))
    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
