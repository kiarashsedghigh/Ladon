"""
Threshold-decapsulation communication cost model for two Lapiha et al. TKEMs.

Paper A  (2025-1958): Lapiha & Prest, "A Lattice-Based IND-CCA Threshold KEM
                      from the BCHK+ Transform" (Asiacrypt'25).
Paper B  (2026-021) : Boudgoust, del Pino, Lapiha & Prest,
                      "IND-CCA Lattice Threshold KEM under 30 KiB" (PKC'26).

Communication model (explicit conventions, in n = #participating servers):

  BROADCAST model           a party broadcasts to the OTHER (n-1) parties:
                              per party    = (n-1) * data
                              system total =  n * (n-1) * data
                              rounds       = 1

  KING (relay) model        every party sends 'data' to one designated party (the
                            king); the king then broadcasts back an AGGREGATE of
                            size 'data' to the other (n-1) parties:
                              per party    = data           (one send to king)
                              system total = (n-1) * data   (to-king phase)
                                           + (n-1) * data   (king broadcasts agg)
                                           = 2 * (n-1) * data
                              rounds       = 2
                            NOTE: this only saves comm when the king can actually
                            AGGREGATE; for the commit step every party needs every
                            raw cmt_j, so the king must rebroadcast all n of them,
                            giving system = (n-1)*cmt + (n-1)*n*cmt = (n^2-1)*cmt.

  3-phase protocol:
    Phase 0   distribute ct      dealer -> n servers ; system = n * |ct|
    Phase 1   threshold decryp.  3 rounds of ShareExtract:
                step 1  send Hcmt(w_i)        ( 2*kappa bits )
                step 2  open  w_i             ( one ring element )
                step 3  send  z_i to combiner ( 4 polys for Paper B / 9 for A )
              Step 3 always uses a single combiner (= "king-with-no-broadcast"),
              so it stays linear in both regimes; the non-round-optimal variant
              additionally has the combiner echo K back to all (+1 round).

|z| = (#polys) * d * ceil(log2 q) / 8  bytes  -- modulus * coeffs-per-poly * 4 polys
      (Paper B: 4 polys.  Paper A: contrib2 = (z,x0,x1) in R^3 x R^3 x R^3 = 9 polys.)

NOTE: |z| polynomials are sized as FULL ring elements -- conservative upper bound.
They are short in reality (bounded by B_ind / B); tightening this just shrinks
step 3's coefficient.
"""
from math import ceil, log2

KiB = 1024


# ---------------------------------------------------------------------------
# Parameter sets taken directly from the papers' tables. key=(paper,kappa,robust)
# ---------------------------------------------------------------------------
PARAMS = {
    # Paper A (Table 2) -- only 128, non-robust given in the paper
    ("A", 128, False): dict(d=4096, log2q=100, N=32,
                            ek_bytes=50 * KiB, ct_tibe_bytes=450 * KiB,
                            sig_vk_bytes=2144 + 64),

    # Paper B (Table 1)
    ("B", 128, False): dict(d=2048, log2q=50, N=32,
                            ek_bytes=6688,  ct_tibe_bytes=28544, sig_vk_bytes=1563),
    ("B", 128, True ): dict(d=2048, log2q=50, N=32,
                            ek_bytes=8224,  ct_tibe_bytes=30368, sig_vk_bytes=1563),
    ("B", 256, False): dict(d=4096, log2q=50, N=32,
                            ek_bytes=13888, ct_tibe_bytes=57056, sig_vk_bytes=3073),
    ("B", 256, True ): dict(d=4096, log2q=50, N=32,
                            ek_bytes=16448, ct_tibe_bytes=61696, sig_vk_bytes=3073),
}

# vector size of the response share z_i (number of polynomials)
Z_RING_ELEMS = {"A": 9, "B": 4}     # see TIBE.ShareExtract2 in each paper


def ring_bytes(d, log2q):
    """One full ring element of R_q: d coefficients * ceil(log2 q) bits / 8."""
    return d * ceil(log2q) / 8.0


# ---------------------------------------------------------------------------
# Polynomial in n, stored as {power: coeff_in_bytes}.
# ---------------------------------------------------------------------------
def poly_add(*polys):
    out = {}
    for p in polys:
        for k, v in p.items():
            out[k] = out.get(k, 0) + v
    return {k: v for k, v in out.items() if v != 0}


def poly_str(p, unit_bytes=KiB, unit_name="KiB"):
    """Pretty-print a polynomial in n, trying to factor common shapes."""
    if not p:
        return f"0 {unit_name}"
    # try to factor common patterns
    keys = sorted(p)
    if keys == [1]:                             # c * n
        c = p[1] / unit_bytes
        return f"{c:,.2f}*n {unit_name}"
    if keys == [0]:                             # constant
        c = p[0] / unit_bytes
        return f"{c:,.2f} {unit_name}"
    if keys == [0, 1] and p[0] == -p[1]:        # c*n - c = c*(n-1)
        c = p[1] / unit_bytes
        return f"{c:,.2f}*(n-1) {unit_name}"
    if keys == [1, 2] and p[1] == -p[2]:        # c*n^2 - c*n = c*n*(n-1)
        c = p[2] / unit_bytes
        return f"{c:,.2f}*n*(n-1) {unit_name}"
    if keys == [0, 2] and p[0] == -p[2]:        # c*n^2 - c = c*(n-1)*(n+1)
        c = p[2] / unit_bytes
        return f"{c:,.2f}*(n-1)*(n+1) {unit_name}"
    # generic expanded form
    terms = []
    for power in sorted(p, reverse=True):
        c = p[power] / unit_bytes
        var = {0: "", 1: "*n", 2: "*n^2"}[power]
        sign = " + " if c >= 0 and terms else (" - " if c < 0 and terms else
                                               ("-" if c < 0 else ""))
        terms.append(f"{sign}{abs(c):,.2f}{var}")
    return "".join(terms) + f" {unit_name}"


# ---------------------------------------------------------------------------
# Cost helpers per opening regime.
# ---------------------------------------------------------------------------
def broadcast_step(data):
    """Every party broadcasts `data` to the (n-1) others, 1 round."""
    return dict(
        per_party={1: data, 0: -data},                # (n-1) * data
        system   ={2: data, 1: -data},                # n * (n-1) * data
        rounds   =1,
    )


def to_single_combiner(data):
    """All n parties send `data` to one combiner: system = n * data, 1 round."""
    return dict(per_party={0: data},
                system   ={1: data},                      # n * data
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
    ct  = p["ct_tibe_bytes"] + p["sig_vk_bytes"]

    # Phase 0: distribute ct
    P0 = dict(per_party={0: ct},       # each server receives ct once
              system   ={1: ct},       # n * ct total
              rounds   =1)

    # Phase 1 sub-steps (round-optimal; everything inter-server is BROADCAST,
    # except the final z_i which goes to a single combiner counted as n * z).
    step1 = broadcast_step(cmt)                  # commit  -> broadcast
    step2 = broadcast_step(w)                    # open w  -> broadcast
    step3 = to_single_combiner(z)                # response z -> combiner (n * z)

    P1_per_party = poly_add(step1["per_party"], step2["per_party"], step3["per_party"])
    P1_system    = poly_add(step1["system"],    step2["system"],    step3["system"])
    P1_rounds    = step1["rounds"] + step2["rounds"] + step3["rounds"]

    sk, sk_share = _sk_sizes(paper, p)
    return dict(
        paper=paper, kappa=kappa, robust=robust,
        sizes=dict(sk=sk, sk_share=sk_share,
                   pk_ek=p["ek_bytes"], ct=ct,
                   z=z, w=w, cmt=cmt),
        P0=P0,
        P1=dict(steps=dict(step1_hash_commit=step1,
                           step2_open_commit_w=step2,
                           step3_response_z=step3),
                total_per_party=P1_per_party,
                total_system=P1_system,
                total_rounds=P1_rounds),
        total_per_party=poly_add(P0["per_party"], P1_per_party),
        total_system   =poly_add(P0["system"],    P1_system),
        total_rounds   =P0["rounds"] + P1_rounds,
    )


def _sk_sizes(paper, p):
    """Return (|sk|, sk_share_repr).

    |sk|           : the full decapsulation key the dealer holds before sharing.
                     Both papers: dk = (s, s') in R^2  ->  2 ring elements.
    sk_share_repr  : a dict {'const': c, 'log': l} representing
                       |sk_share| = c + l * log2(N)  bytes,
                     where N is the total number of parties at deployment time.
                     Paper A : Shamir, |sk_share| = |sk|  (N-independent)
                     Paper B : VSS,    |sk_share| ≈ (log2(N) + 1) · |sk|
                               (APPROX upper bound -- VSS shares are SHORT but
                                a user holds up to log2(N)+1 of them.)
    """
    sk = 2 * ring_bytes(p["d"], p["log2q"])
    if paper == "A":
        sk_share = {"const": sk, "log": 0}            # = |sk|
    else:
        sk_share = {"const": sk, "log": sk}           # = (log2(N) + 1) * |sk|
    return sk, sk_share


def sk_share_str(rep, unit_bytes=KiB, unit_name="KiB"):
    c, l = rep["const"], rep["log"]
    if l == 0:
        return f"{c/unit_bytes:,.1f} {unit_name}"
    if c == l:
        return f"{c/unit_bytes:,.2f}*(log2(N) + 1) {unit_name}"
    return (f"{c/unit_bytes:,.2f} + {l/unit_bytes:,.2f}*log2(N) {unit_name}")


# ---------------------------------------------------------------------------
# Public per-paper entry points.
# ---------------------------------------------------------------------------
def cost_lapiha_prest_1958(kappa):
    return _cost("A", kappa, robust=False)


def cost_under_30kib_2026021(kappa, robust=False):
    return _cost("B", kappa, robust=robust)


# ---------------------------------------------------------------------------
# Printer
# ---------------------------------------------------------------------------
def _fmt(b):
    return f"{b/KiB:,.1f} KiB" if b >= KiB else f"{b:,.0f} B"


def _print_step(label, step):
    pp = poly_str(step["per_party"])
    sy = poly_str(step["system"])
    r  = step["rounds"]
    print(f"        {label}")
    print(f"            per party : {pp}")
    print(f"            system    : {sy}    ({r} round{'s' if r!=1 else ''})")


def show(res, label):
    print(f"\n  -- {label} --")
    if res is None:
        print("     (no parameter set provided in the paper)")
        return
    s = res["sizes"]
    print(f"     regime=round-optimal (broadcast; z_i to combiner)  robust={res['robust']}")
    print(f"     sizes  : |sk|={_fmt(s['sk'])}  |sk_share|={sk_share_str(s['sk_share'])}  "
          f"|pk/ek|={_fmt(s['pk_ek'])}  |ct|={_fmt(s['ct'])}")
    print(f"              |cmt|={_fmt(s['cmt'])}   |w|={_fmt(s['w'])}   "
          f"|z|={_fmt(s['z'])}  (= 4 polys * d * log2 q / 8 for Paper B)")

    _print_step("Phase 0  distribute ct :", res["P0"])

    print(f"        Phase 1  threshold decryption :")
    step_labels = {
        "step1_hash_commit":   "step 1  send Hcmt(w_i) :",
        "step2_open_commit_w": "step 2  open w_i       :",
        "step3_response_z":    "step 3  send z_i       :",
    }
    for sk, lbl in step_labels.items():
        _print_step(lbl, res["P1"]["steps"][sk])
    print(f"            Phase 1 per-party total : {poly_str(res['P1']['total_per_party'])}")
    print(f"            Phase 1 system    total : {poly_str(res['P1']['total_system'])}"
          f"    ({res['P1']['total_rounds']} rounds)")

    print(f"     END-TO-END per-party total : {poly_str(res['total_per_party'])}")
    print(f"     END-TO-END system    total : {poly_str(res['total_system'])}"
          f"    ({res['total_rounds']} rounds)")


PAPER_A_NOTE = """\
Note on how |sk|, |sk_share|, |pk|, |ct| are computed (Paper A: κ=128 only):

   |pk| = |ek|                            = 50 KiB         (Table 2, directly)

   |ct_TKEM| = |ct_TIBE| + |vk| + |sig|
             = 450 KiB + 64 B + 2144 B   ≈ 452.2 KiB
              |ct_TIBE| ............ Table 2
              |vk| ................. seed (2κ bits = 32 B) + vk_1 (32 B) ≈ 64 B  (Fig. 6)
              |sig| ................ WOTS+ = 2144 B                              (Theorem 6)

   |sk| = 2 · |w|        (the FULL decapsulation key the dealer holds before sharing)
        = 2 · d · ⌈log₂ q⌉ / 8  =  2 · 4096 · 100 / 8  =  100 KiB
        Alg. 1 line 2: (s_a, e_a) ← D_{R²,ς_a}  —  two ring elements.

   |sk_share| = |sk| = 100 KiB        (what ONE party stores after sharing)
        Alg. 1 line 9: dk_i = (⟦s_a⟧_i, ⟦e_a⟧_i) — two Shamir shares over R_q.
        Shamir gives no compression: each share is the same size as the secret.
"""

PAPER_B_NOTE = """\
Note on how |sk|, |sk_share|, |pk|, |ct| are computed (Paper B has four rows in Table 1):

   |pk| = |ek|                            directly from Table 1
                                          6 688 B (128,nr) … 16 448 B (256,robust)

   |ct_TKEM| = |ct_TIBE| + |vk| + |sig|
              |ct_TIBE| ............ Table 1: 28 544 B (128,nr) … 61 696 B (256,r)
              |vk|+|sig| ........... §7 last paragraph:
                                       κ=128 → Falcon-512  = 1 563 B
                                       κ=256 → Falcon-1024 = 3 073 B

   |sk| = 2 · |w|        (the FULL decapsulation key the dealer holds before sharing)
        = 2 · d · ⌈log₂ q⌉ / 8     κ=128 → 25 KiB ;  κ=256 → 50 KiB
        Alg. 4 line 2: (s, s') ← D_{R²,σ_s}.

   |sk_share| ≈ (log₂ N + 1) · |sk|       [APPROX upper bound — SYMBOLIC in N]
        Alg. 5 + Lemma 11: VandShare splits (s, s') via Vandermonde Secret
        Sharing.  A user holds up to (log₂ N + 1) shares; each share is SHORT
        (sized here as a full ring elt -> loose upper bound).
        N is a DEPLOYMENT parameter chosen by the dealer at setup, not a
        parameter of the construction.  Constraint: T ≤ N, and for the ROBUST
        variant additionally N ≤ 32 (Lemma 12 requires (log N + 1)·σ_s small).
"""


def report_paper(title, note, configs):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)
    print(note)
    for label, res in configs:
        show(res, label)


if __name__ == "__main__":
    # ---- Paper A first, in full --------------------------------------------
    report_paper("PAPER A  -  Lapiha & Prest, BCHK+ TKEM (2025-1958)",
                 PAPER_A_NOTE, [
                     ("kappa=128",      cost_lapiha_prest_1958(128)),
                     ("kappa=256",      cost_lapiha_prest_1958(256)),
                 ])

    # ---- Paper B second, in full -------------------------------------------
    report_paper("PAPER B  -  'IND-CCA Lattice Threshold KEM under 30 KiB' (2026-021)",
                 PAPER_B_NOTE, [
                     ("kappa=128 non-robust", cost_under_30kib_2026021(128, False)),
                     ("kappa=128 robust",     cost_under_30kib_2026021(128, True)),
                     ("kappa=256 non-robust", cost_under_30kib_2026021(256, False)),
                     ("kappa=256 robust",     cost_under_30kib_2026021(256, True)),
                 ])