"""
Threshold-decapsulation communication cost model for two Lapiha et al. TKEMs.

Paper A  (2025-1958): Lapiha & Prest, "A Lattice-Based IND-CCA Threshold KEM
                      from the BCHK+ Transform" (Asiacrypt'25).
Paper B  (2026-021) : Boudgoust, del Pino, Lapiha & Prest,
                      "IND-CCA Lattice Threshold KEM under 30 KiB" (PKC'26).

We only consider the ROUND-OPTIMAL (broadcast) communication model:
    BROADCAST                 per party = (n-1) * data ,  system = n*(n-1)*data , 1 round
    SEND TO SINGLE COMBINER   per party = data         ,  system = n*data       , 1 round

3-phase protocol:
    Phase 0   distribute ct        owner -> n servers      ; system = n * |ct|
    Phase 1   threshold decryption 3 rounds of ShareExtract:
                step 1  Hcmt(w_i) broadcast
                step 2  open w_i  broadcast
                step 3  z_i sent to a single combiner

PER-PARTY end-to-end EXCLUDES Phase 0 (server passively receives ct from the owner).
SYSTEM    end-to-end INCLUDES  Phase 0 (n * |ct| is real network traffic).
"""
from math import ceil, log2

KiB = 1024
GREEN = "\033[92m"
RESET = "\033[0m"


def _both(b):
    """Format a byte count as 'X B (Y KiB)' with KiB highlighted in green."""
    b_str = f"{int(b):>10,} B" if b == int(b) else f"{b:>10,.0f} B"
    k_str = f"{b/KiB:,.2f} KiB"
    return f"{b_str}  ({GREEN}{k_str}{RESET})"


# ---------------------------------------------------------------------------
# Parameter sets taken directly from the papers' tables. key=(paper,kappa,robust)
# ---------------------------------------------------------------------------
PARAMS = {
    # ---- Paper A : Table 2 (only kappa=128, non-robust given) ---------------
    ("A", 128, False): dict(
        d=4096, log2q=100, N=32,
        ek_bytes      = 50 * KiB,    # Table 2  -> |ek|
        ct_tibe_bytes = 450 * KiB,   # Table 2  -> |ct_TIBE|
        vk_bytes      = 64,          # Fig. 6   -> seed (2k bits = 32 B) + vk_1 (32 B)
        sig_bytes     = 2144,        # Thm 6    -> WOTS+ one-time signature
        sig_scheme    = "WOTS+",
    ),
    # ---- Paper B : Table 1 (4 rows) -----------------------------------------
    ("B", 128, False): dict(
        d=2048, log2q=50, N=32,
        ek_bytes      = 6688,        # Table 1  row kappa=128 non-robust
        ct_tibe_bytes = 28544,
        sig_vk_bytes  = 1563,        # Section 7  -> Falcon-512 |vk|+|sig|
        sig_scheme    = "Falcon-512",
    ),
    ("B", 128, True): dict(
        d=2048, log2q=50, N=32,
        ek_bytes      = 8224,        # Table 1  row kappa=128 robust
        ct_tibe_bytes = 30368,
        sig_vk_bytes  = 1563,
        sig_scheme    = "Falcon-512",
    ),
    ("B", 256, False): dict(
        d=4096, log2q=50, N=32,
        ek_bytes      = 13888,       # Table 1  row kappa=256 non-robust
        ct_tibe_bytes = 57056,
        sig_vk_bytes  = 3073,        # Section 7  -> Falcon-1024 |vk|+|sig|
        sig_scheme    = "Falcon-1024",
    ),
    ("B", 256, True): dict(
        d=4096, log2q=50, N=32,
        ek_bytes      = 16448,       # Table 1  row kappa=256 robust
        ct_tibe_bytes = 61696,
        sig_vk_bytes  = 3073,
        sig_scheme    = "Falcon-1024",
    ),
}

# vector size of the response share z_i (number of polynomials)
Z_RING_ELEMS = {"A": 9, "B": 4}


def ring_bytes(d, log2q):
    return d * ceil(log2q) / 8.0


# ---------------------------------------------------------------------------
# Polynomial in n.
# ---------------------------------------------------------------------------
def poly_add(*polys):
    out = {}
    for p in polys:
        for k, v in p.items():
            out[k] = out.get(k, 0) + v
    return {k: v for k, v in out.items() if v != 0}


def poly_str(p, unit_bytes=KiB, unit_name="KiB"):
    if not p:
        return f"0 {unit_name}"
    keys = sorted(p)
    if keys == [1]:                          return f"{p[1]/unit_bytes:,.2f}*n {unit_name}"
    if keys == [0]:                          return f"{p[0]/unit_bytes:,.2f} {unit_name}"
    if keys == [0,1] and p[0] == -p[1]:      return f"{p[1]/unit_bytes:,.2f}*(n-1) {unit_name}"
    if keys == [1,2] and p[1] == -p[2]:      return f"{p[2]/unit_bytes:,.2f}*n*(n-1) {unit_name}"
    terms = []
    for power in sorted(p, reverse=True):
        c = p[power] / unit_bytes
        var = {0: "", 1: "*n", 2: "*n^2"}[power]
        sign = " + " if c >= 0 and terms else (" - " if c < 0 and terms else
                                               ("-" if c < 0 else ""))
        terms.append(f"{sign}{abs(c):,.2f}{var}")
    return "".join(terms) + f" {unit_name}"


def poly_str_both(p):
    """Return 'poly_in_B  (poly_in_KiB)' with KiB part green."""
    b = poly_str(p, unit_bytes=1, unit_name="B")
    k = poly_str(p, unit_bytes=KiB, unit_name="KiB")
    return f"{b}    ({GREEN}{k}{RESET})"


# ---------------------------------------------------------------------------
# Cost helpers (broadcast model only).
# ---------------------------------------------------------------------------
def broadcast_step(data):
    return dict(per_party={1: data, 0: -data},
                system   ={2: data, 1: -data},
                rounds   =1)


def to_single_combiner(data):
    return dict(per_party={0: data},
                system   ={1: data},
                rounds   =1)


# ---------------------------------------------------------------------------
# Core cost computation.
# ---------------------------------------------------------------------------
def _cost(paper, kappa, robust):
    key = (paper, kappa, robust)
    if key not in PARAMS:
        return None
    p = PARAMS[key]
    cmt = 2 * kappa / 8.0
    w   = ring_bytes(p["d"], p["log2q"])
    z   = Z_RING_ELEMS[paper] * w
    # |ct| = |ct_TIBE| + |vk| + |sig|; the two sig schemes are bundled differently
    sig_vk = p.get("sig_vk_bytes",
                   p.get("sig_bytes", 0) + p.get("vk_bytes", 0))
    ct  = p["ct_tibe_bytes"] + sig_vk

    P0    = dict(per_party={0: ct}, system={1: ct}, rounds=1)
    step1 = broadcast_step(cmt)
    step2 = broadcast_step(w)
    step3 = to_single_combiner(z)
    P1_per_party = poly_add(step1["per_party"], step2["per_party"], step3["per_party"])
    P1_system    = poly_add(step1["system"],    step2["system"],    step3["system"])

    sk, sk_share = _sk_sizes(paper, p)
    return dict(
        paper=paper, kappa=kappa, robust=robust,
        sizes=dict(sk=sk, sk_share=sk_share,
                   pk_ek=p["ek_bytes"], ct=ct,
                   z=z, w=w, cmt=cmt),
        P0=P0,
        P1=dict(steps=dict(step1=step1, step2=step2, step3=step3),
                total_per_party=P1_per_party,
                total_system=P1_system,
                total_rounds=step1["rounds"] + step2["rounds"] + step3["rounds"]),
        total_per_party_excl_P0=P1_per_party,
        total_per_party_incl_P0=poly_add(P0["per_party"], P1_per_party),
        total_system           =poly_add(P0["system"],    P1_system),
        total_rounds           =P0["rounds"] + step1["rounds"] + step2["rounds"] + step3["rounds"],
    )


def _sk_sizes(paper, p):
    sk = 2 * ring_bytes(p["d"], p["log2q"])
    if paper == "A":
        sk_share = {"const": sk, "log": 0}
    else:
        sk_share = {"const": sk, "log": sk}
    return sk, sk_share


def sk_share_str(rep, unit_bytes=KiB, unit_name="KiB"):
    c, l = rep["const"], rep["log"]
    if l == 0:
        return f"{c/unit_bytes:,.1f} {unit_name}"
    if c == l:
        return f"{c/unit_bytes:,.2f}*(log2(N) + 1) {unit_name}"
    return f"{c/unit_bytes:,.2f} + {l/unit_bytes:,.2f}*log2(N) {unit_name}"


# ---------------------------------------------------------------------------
# Public entry points.
# ---------------------------------------------------------------------------
def cost_lapiha_prest_1958(kappa):       return _cost("A", kappa, robust=False)
def cost_under_30kib_2026021(kappa, robust=False):  return _cost("B", kappa, robust=robust)


# ---------------------------------------------------------------------------
# Printer with per-config inline derivations.
# ---------------------------------------------------------------------------
def _fmt(b):
    if b == int(b) and b < 10_000: return f"{int(b):,} B"
    return f"{b/KiB:,.1f} KiB" if b >= KiB else f"{b:,.0f} B"


def _print_derivation(res):
    paper = res["paper"]; kappa = res["kappa"]; robust = res["robust"]
    p = PARAMS[(paper, kappa, robust)]
    s = res["sizes"]
    d, lq = p["d"], p["log2q"]
    sk_bytes = 2 * ring_bytes(d, lq)

    print("     --- Sizes (with derivation) ---")

    if paper == "A":
        src = "Table 2"
    else:
        src = f"Table 1, row kappa={kappa} {'robust' if robust else 'non-robust'}"

    # |pk|
    print(f"     |pk| = |ek|     = {_both(p['ek_bytes'])}    [{src}]")

    # |sk|
    print(f"     |sk| = 2 * |w|  = 2 * d * ceil(log2 q) / 8 = 2 * {d} * {lq} / 8")
    print(f"                     = {_both(sk_bytes)}")
    src_sk = "Alg. 1 line 2: (s_a, e_a) in R^2" if paper == "A" \
        else "Alg. 4 line 2: (s, s') in R^2"
    print(f"                       [{src_sk}]")

    # |sk_share|
    if paper == "A":
        print(f"     |sk_share| = |sk| = {_both(sk_bytes)}    "
              f"[Shamir, Alg. 1 line 9: no compression]")
    else:
        print(f"     |sk_share| ~ (log2 N + 1) * |sk|")
        print(f"                = {sk_share_str(s['sk_share'])}    "
              f"[VSS, Alg. 5 + Lemma 11; APPROX upper bound]")

    # |ct|
    print(f"     |ct| = |ct_TIBE|  +  |vk|+|sig|")
    print(f"          |ct_TIBE| = {_both(p['ct_tibe_bytes'])}    [{src}]")
    if paper == "A":
        print(f"          |vk|      = {_both(p['vk_bytes'])}    "
              f"[Fig. 6: seed 2*kappa bits ({p['vk_bytes']//2} B) + vk_1 ({p['vk_bytes']//2} B)]")
        print(f"          |sig|     = {_both(p['sig_bytes'])}    "
              f"[Theorem 6: {p['sig_scheme']} one-time sig]")
    else:
        print(f"          |vk|+|sig|= {_both(p['sig_vk_bytes'])}    "
              f"[Section 7: {p['sig_scheme']} |vk|+|sig|]")
    print(f"          --------")
    print(f"          TOTAL     = {_both(s['ct'])}")

    # protocol message sizes
    print("     --- Protocol message sizes ---")
    print(f"     |cmt| = 2*kappa/8           = {_both(s['cmt'])}    "
          f"[Hcmt: R_q -> {{0,1}}^(2*kappa)]")
    print(f"     |w|   = d * ceil(log2 q) /8 = {_both(s['w'])}    "
          f"[one ring element of R_q]")
    z_polys = Z_RING_ELEMS[paper]
    z_src = ("contrib2 = (z, x_0, x_1) in R^3 x R^3 x R^3, Alg. 7" if paper == "A"
             else "z^(i) in R^4, Alg. 12")
    print(f"     |z|   = {z_polys} * |w|              = {_both(s['z'])}    [{z_src}]")


def _print_step(label, step):
    print(f"        {label}")
    print(f"            per party : {poly_str_both(step['per_party'])}")
    print(f"            system    : {poly_str_both(step['system'])}    "
          f"({step['rounds']} round{'s' if step['rounds']!=1 else ''})")


def show(res, label):
    print(f"\n  ============ {label} ============")
    if res is None:
        print("     (no parameter set provided in the paper)")
        return
    print(f"     regime : round-optimal (broadcast; z_i to single combiner)   "
          f"robust={res['robust']}")
    _print_derivation(res)

    print("     --- Communication ---")
    _print_step("Phase 0  distribute ct :", res["P0"])
    print(f"        Phase 1  threshold decryption :")
    _print_step("step 1  send Hcmt(w_i) :", res["P1"]["steps"]["step1"])
    _print_step("step 2  open w_i       :", res["P1"]["steps"]["step2"])
    _print_step("step 3  send z_i       :", res["P1"]["steps"]["step3"])
    print(f"            Phase 1 per-party total : {poly_str_both(res['P1']['total_per_party'])}")
    print(f"            Phase 1 system    total : {poly_str_both(res['P1']['total_system'])}"
          f"    ({res['P1']['total_rounds']} rounds)")
    print(f"     END-TO-END per-party total (excl. Phase 0) : "
          f"{poly_str_both(res['total_per_party_excl_P0'])}")
    print(f"     END-TO-END per-party total (incl. Phase 0) : "
          f"{poly_str_both(res['total_per_party_incl_P0'])}")
    print(f"     END-TO-END system    total                 : "
          f"{poly_str_both(res['total_system'])}    ({res['total_rounds']} rounds)")


def report_paper(title, configs):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)
    for label, res in configs:
        show(res, label)


if __name__ == "__main__":
    report_paper("PAPER A  -  Lapiha & Prest, BCHK+ TKEM (2025-1958)", [
        ("kappa=128", cost_lapiha_prest_1958(128)),
        ("kappa=256", cost_lapiha_prest_1958(256)),
    ])
    report_paper("PAPER B  -  'IND-CCA Lattice Threshold KEM under 30 KiB' (2026-021)", [
        ("kappa=128 non-robust", cost_under_30kib_2026021(128, False)),
        ("kappa=128 robust",     cost_under_30kib_2026021(128, True)),
        ("kappa=256 non-robust", cost_under_30kib_2026021(256, False)),
        ("kappa=256 robust",     cost_under_30kib_2026021(256, True)),
    ])