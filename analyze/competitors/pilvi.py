"""
Threshold-decryption communication cost model for Pilvi.

Paper : Cini, Lai & Woo, "Pilvi: Lattice Threshold PKE with Small Decryption
        Shares and Improved Security" (eprint 2025/1691).

================================================================================
SCHEME STRUCTURE (Pilvi = thresholdised Regev PKE, Fig. 5)
================================================================================
Section 7 fixes:  f = 512  ->  phi = euler_phi(512) = 256
                  secpar   = 128
                  message  = 256 bits  ->  L = ceil(2*secpar/phi) = 1
                  m        = 2n + L

Pilvi is NON-INTERACTIVE: each of the t participating parties computes one
partial decryption pd_k locally and sends it to a single combiner.  No commit,
no open, no inter-party messaging.

   Phase 0  distribute ctxt :  system = n * |ctxt|       (1 round)
   Phase 1  collect shares  :  system = n * |pd_k|       (1 round)

================================================================================
THE FIVE SIZES, FROM Fig. 5
================================================================================
   |w|     = phi * log_2(q) / 8                                bytes / R_q elt

   |ctxt|  = (n + L) * |w|              Fig. 5 Enc:
                                        ctxt = (c_0, (c_{l,1})_l)
                                        c_0 in R_q^n,  c_{l,1} in R_q.

   |pd_k|  = L * |w|                    Fig. 5 ParDec:
                                        pd_k = (s_{l,k}^T c_0 + e_{l,k})_l
                                        -> L ring elements (the "small share").

   |sk|    = L * n * |w|                Fig. 5 KGen:
                                        master sk = (r_l)_l,  r_l in R_q^n.

   |sk_k|  = L * n * |w|                Fig. 5 KGen:
                                        share sk_k = (s_{l,k})_l,
                                        s_{l,k} = (V*R_l)_k in R_q^n.
                                        Shamir over rings, N-independent.

   |pk|    = (seed_A, (b_l)_l)          Fig. 5 KGen, *fairly* counted:
              A    in R_q^{n x m}        derived from a SEED_BYTES-byte seed
                                         via an XOF (standard practice -- Kyber,
                                         Dilithium, FrodoKEM, ...).
                                         The seed lives in pp; not charged per-key.
              b_l  in R_q^m              L*m * |w|   (fresh per KGen)
            |pk|       = SEED_BYTES + L*m * |w|       <- the fair number
            |pk_dense| = (n + L)*m * |w|              <- without the seed trick
            |pk_A|     = n*m * |w|                    <- A expanded (informational)

================================================================================
HOW TABLE 5 IS REPRODUCED (subtle point)
================================================================================
The paper's size formula uses the CONTINUOUS log_2(q), while Table 5's
modulus column reports ceil(log_2 q).  Back-solving Table 5:

      log_2(q) = (|ctxt|_KiB * 1024 * 8) / ((n+L) * phi)

E.g. Pilvi2048-6-8-1 :  17.5 * 1024 * 8 / (9 * 256) = 62.222 ,
     and  |pd_k| = 256 * 62.222 / 8 = 1991 B ~ 1.9 KiB.

So q ~ 2^62.22 in reality; Table 5 just writes "2^63" (= ceil).

The PARAMS table below stores BOTH:
     log2q       = continuous log_2(q), reproduces |ctxt| / |pd_k| to last digit
     ceil_log2q  = the integer printed in Table 5's modulus column

A real implementation would pack each coefficient in ceil(log_2 q) bits, so the
practical wire-size is slightly larger than the paper's reported number.
"""
from math import ceil, log2

KiB = 1024
GREEN = "\033[92m"
RESET = "\033[0m"

PHI    = 256              # euler_phi(512)
SECPAR = 128
L_MSG  = 1                # ceil(2*128 / 256)

# A is sampled from a public seed via an XOF (standard in lattice crypto).
# 32 bytes = 256 bits, matching the security level lambda = 128 with margin.
SEED_BYTES = 32

# ---------------------------------------------------------------------------
# Parameter sets ("This work" rows of Table 5).
#   log2q       : continuous log_2(q)  (reproduces |ctxt|, |pd_k| in KiB)
#   ceil_log2q  : ceil(log_2 q)        (the value printed in Table 5)
# ---------------------------------------------------------------------------
PARAMS = {
    # name                       n   log2q    ceil  t   K    Q
    "Pilvi1792-2-8-1":     dict(n=7,  log2q=56.000,  ceil_log2q=56,  t=2,  K=8,  Q=1),
    "Pilvi2048-6-8-1":     dict(n=8,  log2q=62.222,  ceil_log2q=63,  t=6,  K=8,  Q=1),
    "Pilvi2304-10-16-1":   dict(n=9,  log2q=69.760,  ceil_log2q=70,  t=10, K=16, Q=1),
    "Pilvi2816-16-32-1":   dict(n=11, log2q=82.933,  ceil_log2q=83,  t=16, K=32, Q=1),
    "Pilvi3072-2-8-x60":   dict(n=12, log2q=88.123,  ceil_log2q=89,  t=2,  K=8,  Q=2**60),
    "Pilvi3072-6-8-x60":   dict(n=12, log2q=93.785,  ceil_log2q=94,  t=6,  K=8,  Q=2**60),
    "Pilvi3584-10-16-x60": dict(n=14, log2q=101.547, ceil_log2q=102, t=10, K=16, Q=2**60),
    "Pilvi3840-16-32-x60": dict(n=15, log2q=114.200, ceil_log2q=115, t=16, K=32, Q=2**60),
}

# Paper Table 5 ("This work" rows) - for the self-check at the bottom.
TABLE5 = {
    # name                  |ctxt| KB,  |pd_k| KB
    "Pilvi1792-2-8-1":     (14.0, 1.7),
    "Pilvi2048-6-8-1":     (17.5, 1.9),
    "Pilvi2304-10-16-1":   (21.8, 2.2),
    "Pilvi2816-16-32-1":   (31.1, 2.6),
    "Pilvi3072-2-8-x60":   (35.8, 2.8),
    "Pilvi3072-6-8-x60":   (38.1, 2.9),
    "Pilvi3584-10-16-x60": (47.6, 3.2),
    "Pilvi3840-16-32-x60": (57.1, 3.6),
}


# ---------------------------------------------------------------------------
# Core size formulas
# ---------------------------------------------------------------------------
def ring_bytes(phi, log2q):
    """One element of R_q in bytes.  phi coefficients, log_2(q) bits each.

    Uses the *continuous* log_2(q) -- this is the paper's reported size and
    matches Table 5 exactly.  A real implementation needs ceil(log_2 q) bits
    per coefficient, giving slightly larger ciphertexts."""
    return phi * log2q / 8.0


def all_sizes(name):
    """Compute every size from Fig. 5 for a given parameter set."""
    p = PARAMS[name]
    phi, L = PHI, L_MSG
    n      = p["n"]
    m      = 2 * n + L                  # Section 7: m = 2n + L
    log2q  = p["log2q"]

    w        = ring_bytes(phi, log2q)
    ctxt     = (n + L) * w              # Fig. 5 Enc
    pd_k     = L * w                    # Fig. 5 ParDec
    sk_mst   = L * n * w                # master  sk = (r_l)_l ,  r_l in R_q^n
    sk_share = L * n * w                # share   s_{l,k} in R_q^n (one row)
    pk_A     = n * m * w                # A expanded (informational only)
    pk_b     = L * m * w                # (b_l)_l, each in R_q^m  (fresh per KGen)
    pk       = SEED_BYTES + pk_b        # FAIR: A from seed, only b's are new
    pk_dense = pk_A + pk_b              # without the seed trick (Fig. 5 verbatim)

    return dict(
        name=name, n=n, m=m, t=p["t"], K=p["K"], Q=p["Q"],
        log2q=log2q, ceil_log2q=p["ceil_log2q"],
        w=w, ctxt=ctxt, pd_k=pd_k,
        sk_master=sk_mst, sk_share=sk_share,
        pk_A=pk_A, pk_b=pk_b, pk=pk, pk_dense=pk_dense,
        seed=SEED_BYTES,
    )


# ---------------------------------------------------------------------------
# Polynomial-in-n helpers (kept from your original script)
# ---------------------------------------------------------------------------
def poly_add(*polys):
    out = {}
    for p in polys:
        for k, v in p.items():
            out[k] = out.get(k, 0) + v
    return {k: v for k, v in out.items() if v != 0}


def poly_eval(p, n):
    return sum(c * (n ** power) for power, c in p.items())


def _poly_core(p, unit_bytes, fmt):
    if not p:
        return "0"
    terms = []
    for power in sorted(p, reverse=True):
        cc = p[power] / unit_bytes
        var = {0: "", 1: "*n", 2: "*n^2"}[power]
        sign = " + " if cc >= 0 and terms else (" - " if cc < 0 and terms else
                                                ("-" if cc < 0 else ""))
        terms.append(f"{sign}{fmt(abs(cc))}{var}")
    return "".join(terms)


def poly_str(p):
    bytes_part = _poly_core(p, 1,   lambda x: f"{x:,.2f}")
    kib_part   = _poly_core(p, KiB, lambda x: f"{x:,.2f}")
    return f"{bytes_part} B    ({GREEN}{kib_part} KiB{RESET})"


def to_single_combiner(data):
    return dict(per_party={0: data},
                system   ={1: data},
                rounds   =1)


# ---------------------------------------------------------------------------
# Communication cost (non-interactive: 1 send per party, n*pd in system)
# ---------------------------------------------------------------------------
def cost_pilvi(name):
    if name not in PARAMS:
        return None
    s   = all_sizes(name)
    P0  = dict(per_party={0: s["ctxt"]}, system={1: s["ctxt"]}, rounds=1)
    P1  = to_single_combiner(s["pd_k"])
    return dict(
        name=name, sizes=s,
        P0=P0, P1=dict(step=P1, total_per_party=P1["per_party"],
                       total_system=P1["system"], total_rounds=P1["rounds"]),
        total_per_party_excl_P0=P1["per_party"],
        total_per_party_incl_P0=poly_add(P0["per_party"], P1["per_party"]),
        total_system           =poly_add(P0["system"],    P1["system"]),
        total_rounds           =P0["rounds"] + P1["rounds"],
    )


# ---------------------------------------------------------------------------
# Pretty-printing helpers
# ---------------------------------------------------------------------------
def _both(b):
    if b is None:
        return "n/a"
    b_str = f"{b:,.0f} B"
    if b >= KiB * KiB:
        k_str = f"{b/(KiB*KiB):,.2f} MiB"
    else:
        k_str = f"{b/KiB:,.2f} KiB" if b >= KiB else f"{b:,.2f} B"
    return f"{b_str}    ({GREEN}{k_str}{RESET})"


def _plain(b):
    if b is None:
        return "n/a"
    if b >= KiB * KiB:
        return f"{b/(KiB*KiB):,.2f} MiB"
    return f"{b/KiB:,.2f} KiB" if b >= KiB else f"{b:,.0f} B"


def _print_step(label, step, n_concrete=None):
    print(f"        {label}")
    print(f"            per party : {poly_str(step['per_party'])}")
    line = (f"            system    : {poly_str(step['system'])}    "
            f"({step['rounds']} round{'s' if step['rounds']!=1 else ''})")
    if n_concrete is not None:
        line += f"   |  at n={n_concrete}: {_plain(poly_eval(step['system'], n_concrete))}"
    print(line)


def show(res):
    s = res["sizes"]
    Qs = f"2^{int(round(log2(s['Q'])))}" if s["Q"] >= 2 else str(s["Q"])
    print(f"\n  ============ {res['name']} ============")
    print(f"     params : t={s['t']}, K={s['K']}, Q={Qs}, "
          f"n={s['n']}, m=2n+L={s['m']}, "
          f"log_2 q = {s['log2q']:.3f}  (Table 5 writes 2^{s['ceil_log2q']})")
    print(f"     --- Sizes (Fig. 5) ---")
    print(f"     |w|         = {_both(s['w'])}              "
          f"one R_q element  =  phi * log_2(q) / 8")
    print(f"     |ctxt|      = {_both(s['ctxt'])}    "
          f"= (n + L) * |w|")
    print(f"     |pd_k|      = {_both(s['pd_k'])}    "
          f"= L * |w|   (small share)")
    print(f"     |sk| master = {_both(s['sk_master'])}    "
          f"= L * n * |w|")
    print(f"     |sk_k| share= {_both(s['sk_share'])}    "
          f"= L * n * |w|   (Shamir, N-independent)")
    print(f"     |pk|        = {_both(s['pk'])}    "
          f"= {SEED_BYTES} B (seed for A) + L*m*|w|   (FAIR: A from XOF)")
    print(f"       . pk_b     = {_both(s['pk_b'])}    "
          f"= L * m * |w|   (the L 'b_l' vectors -- fresh per KGen)")
    print(f"       . seed_A   = {_both(s['seed'])}    "
          f"= {SEED_BYTES} B   (A reproducible by anyone)")
    print(f"     |pk_dense|  = {_both(s['pk_dense'])}    "
          f"= (n + L)*m*|w|   (A expanded, Fig. 5 verbatim)")
    print(f"       . pk_A     = {_both(s['pk_A'])}    "
          f"= n * m * |w|   (informational; replaced by seed above)")

    print(f"     --- Communication ---")
    nt = s["t"]
    _print_step("Phase 0  distribute ctxt :", res["P0"], n_concrete=nt)
    print(f"        Phase 1  collect pd_k (NON-INTERACTIVE, 1 send/party):")
    _print_step("step    pd_k -> combiner :", res["P1"]["step"], n_concrete=nt)

    print(f"     END-TO-END per-party total (excl. P0) : "
          f"{poly_str(res['total_per_party_excl_P0'])}")
    print(f"     END-TO-END per-party total (incl. P0) : "
          f"{poly_str(res['total_per_party_incl_P0'])}")
    print(f"     END-TO-END system    total            : "
          f"{poly_str(res['total_system'])}    ({res['total_rounds']} rounds)")
    print(f"     --- At n = t = {nt} (decryption threshold) ---")
    print(f"        per-party (excl. P0) : "
          f"{_plain(poly_eval(res['total_per_party_excl_P0'], nt))}")
    print(f"        per-party (incl. P0) : "
          f"{_plain(poly_eval(res['total_per_party_incl_P0'], nt))}")
    print(f"        system               : "
          f"{_plain(poly_eval(res['total_system'], nt))}")
# ---------------------------------------------------------------------------
# Table 5 self-check
# ---------------------------------------------------------------------------
def verify_table5():
    """Print computed |ctxt|, |pd_k| next to the paper's Table 5 values."""
    print("\n" + "=" * 78)
    print("TABLE 5 SELF-CHECK  (computed  vs  paper)")
    print("=" * 78)
    hdr = (f"{'parameter set':<24}  "
           f"{'|ctxt| comp':>11} {'paper':>7}  "
           f"{'|pd_k| comp':>11} {'paper':>7}  ok?")
    print(hdr)
    print("-" * len(hdr))
    all_ok = True
    for name in PARAMS:
        s = all_sizes(name)
        ctxt_comp = s["ctxt"] / KiB
        pd_comp   = s["pd_k"] / KiB
        ctxt_pap, pd_pap = TABLE5[name]
        # Tolerance = 0.1 KB matches the table's 1-decimal precision
        # (e.g. 1.75 KiB rounds to 1.7 or 1.8 depending on convention).
        ok_c = abs(ctxt_comp - ctxt_pap) <= 0.1
        ok_p = abs(pd_comp   - pd_pap)   <= 0.1
        ok   = ok_c and ok_p
        all_ok &= ok
        mark = "OK" if ok else "MISS"
        print(f"{name:<24}  "
              f"{ctxt_comp:>9.2f} KB {ctxt_pap:>5.1f} KB  "
              f"{pd_comp:>9.2f} KB {pd_pap:>5.1f} KB  {mark}")
    print("-" * len(hdr))
    print("ALL MATCH" if all_ok else "some discrepancies (> 0.05 KB)")


# ---------------------------------------------------------------------------
# Sanity walk-through for one entry (printed when the script is run)
# ---------------------------------------------------------------------------
WORKED_EXAMPLE = """\
Worked example -- Pilvi2048-6-8-1  (t=6, K=8, Q=1):

   phi   = euler_phi(512) = 256
   L     = ceil(2*128 / 256) = 1
   n     = 8 ,  m = 2n + L = 17
   log_2(q) = 62.222    (Table 5 writes 2^63 = ceil)

   |w|     = 256 * 62.222 / 8                    = 1990.99  B
   |ctxt|  = (n + L) * |w| = 9 * 1990.99         = 17918.9  B  = 17.50 KiB  (Table: 17.5)
   |pd_k|  = L * |w|       = 1 * 1990.99         =  1990.99 B  =  1.94 KiB  (Table: 1.9)
   |sk|    = L * n * |w|   = 8 * 1990.99         = 15927.9  B  = 15.55 KiB
   |sk_k|  = L * n * |w|                         = 15927.9  B  = 15.55 KiB  (no compression)
   |pk_b|  = L * m * |w|   = 17 * 1990.99        = 33846.8  B  =  33.06 KiB
   |pk|    = 32 + L*m*|w| = 32 + 33846.8         = 33878.8  B  =  33.09 KiB  (A from seed)
   |pk_A|  = n * m * |w|   = 8 * 17 * 1990.99    = 270774   B  = 264.4  KiB  (only if A is materialised)
   |pk_dense| = (n+L)*m*|w| = 9 * 17 * 1990.99   = 304621   B  = 297.5  KiB  (Fig. 5 verbatim)
"""


def report():
    print("\n" + "=" * 78)
    print("PILVI - Cini, Lai & Woo, Lattice Threshold PKE (eprint 2025/1691)")
    print("=" * 78)
    print(WORKED_EXAMPLE)
    verify_table5()
    for name in PARAMS:
        show(cost_pilvi(name))


if __name__ == "__main__":
    report()