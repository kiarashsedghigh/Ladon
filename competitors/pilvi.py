"""
Threshold-decryption communication cost model for Pilvi.

Paper : Cini, Lai & Woo, "Pilvi: Lattice Threshold PKE with Small Decryption
        Shares and Improved Security" (2025-1691).

================================================================================
HOW THIS DIFFERS FROM THE sasha_prest.py MODEL (and why)
================================================================================
The Lapiha-Prest papers (sasha_prest.py) are INTERACTIVE threshold KEMs built on
a (T)IBE + BCHK+ transform.  Their ShareExtract is a 3-round protocol:
        step 1  send  Hcmt(w_i)     (commit)
        step 2  open  w_i           (reveal)
        step 3  send  z_i           (response to combiner)

Pilvi is a NON-INTERACTIVE (t,K)-threshold PKE (a thresholdised Regev scheme):

    "Our scheme allows non-interactive decryption: no collaboration between
     decrypting parties is required and the plaintext can be recovered by anyone
     collecting sufficiently many partial decryption shares."

So there is NO commit step, NO open step, and NO inter-party messaging.  Each of
the t participating parties locally computes one partial decryption pd_k and
sends it (once) to whoever is reconstructing.  Phase 1 is therefore a SINGLE
message of size |pd_k| per party, in ONE round.

That is the entire "rounds" difference: 3 interactive rounds  ->  1 send.

================================================================================
COMMUNICATION MODEL (explicit conventions; n = #participating parties)
================================================================================
For Pilvi, decryption needs exactly t shares, so the natural setting is n = t.
We keep n symbolic (as in sasha_prest.py) so the per-party / system polynomials
are directly comparable, and also print concrete numbers at n = t.

  Phase 0   distribute ctxt    encryptor -> n parties ; system = n * |ctxt|   (1 round)

  Phase 1   collect shares     each party -> reconstructor, ONE message |pd_k|:

    COMBINER regime (round-optimal, the intended mode):
        every party sends pd_k to ONE combiner who reconstructs the plaintext.
            per party    = |pd_k|              (one send)
            system total = (n-1) * |pd_k|      (combiner needs no self-send)
            rounds       = 1
        Optional echo: the combiner broadcasts the recovered plaintext back to
        the (n-1) other parties (+1 round, + (n-1)*|msg|).  |msg| is tiny.

    BROADCAST regime (every party reconstructs locally, no trusted combiner):
            per party    = (n-1) * |pd_k|
            system total = n*(n-1) * |pd_k|
            rounds       = 1

================================================================================
SIZE FORMULAS (from the Pilvi construction, Fig. 5; phi = euler_phi(f))
================================================================================
  one ring element of R_q   :  |w| = phi * ceil(log2 q) / 8  bytes
  ciphertext ctxt=(c0,(c_l)) :  c0 in R_q^n , c_l in R_q (L of them)
                                |ctxt| = (n + L) * |w|              # = (n+L) phi log q
  partial decryption pd_k     :  (pd_{l,k})_{l in [L]} in R_q^L
                                |pd_k| = L * |w|                    # = L phi log q
  master secret sk = (r_l)    :  L vectors in R_q^n
                                |sk| = L * n * |w|
  key share sk_k = (s_{l,k})  :  L vectors in R_q^n  (Shamir over the subtractive
                                set; one row of V*R_l per l).  No compression and
                                N-INDEPENDENT:  |sk_share| = |sk| = L * n * |w|.

These reproduce the paper's |ctxt| = (n+L) phi log q and |pd_k| = L phi log q,
and Tables 2-5 (|ctxt| 14-58 KB, |pd_k| ~1-4 KB).
"""
from math import ceil

KiB = 1024


# ---------------------------------------------------------------------------
# Parameter sets.  The 8 NAMED sets are taken from Table 5 ("This work" rows).
# The name Pilvi{phi*n}-{t}-{K}-{Qind} encodes the LWE dimension phi*n, the
# recovery threshold t, the max users K, and the query regime.  phi = 256 (f=512),
# L = ceil(2*secpar/phi) = 1 for a 256-bit message.
#   log2q is recovered from the published |ctxt| via |ctxt| = (n+L) phi log q / 8.
# ---------------------------------------------------------------------------
PHI = 256          # euler_phi(512)
SECPAR = 128
L_MSG = 1          # ceil(2*128 / 256)

PARAMS = {
    # name                 n   log2q  t   K     Q
    "Pilvi1792-2-8-1":   dict(n=7,  log2q=56,  t=2,  K=8,  Q=1),
    "Pilvi2048-6-8-1":   dict(n=8,  log2q=62,  t=6,  K=8,  Q=1),
    "Pilvi2304-10-16-1": dict(n=9,  log2q=70,  t=10, K=16, Q=1),
    "Pilvi2816-16-32-1": dict(n=11, log2q=83,  t=16, K=32, Q=1),
    "Pilvi3072-2-8-x60": dict(n=12, log2q=89,  t=2,  K=8,  Q=2**60),
    "Pilvi3072-6-8-x60": dict(n=12, log2q=94,  t=6,  K=8,  Q=2**60),
    "Pilvi3584-10-16-x60": dict(n=14, log2q=102, t=10, K=16, Q=2**60),
    "Pilvi3840-16-32-x60": dict(n=15, log2q=115, t=16, K=32, Q=2**60),
}


def ring_bytes(phi, log2q):
    """One full ring element of R_q: phi coefficients * ceil(log2 q) bits / 8."""
    return phi * ceil(log2q) / 8.0


# ---------------------------------------------------------------------------
# Polynomial in n, stored as {power: coeff_in_bytes}.   (identical to reference)
# ---------------------------------------------------------------------------
def poly_add(*polys):
    out = {}
    for p in polys:
        for k, v in p.items():
            out[k] = out.get(k, 0) + v
    return {k: v for k, v in out.items() if v != 0}


def poly_eval(p, n):
    return sum(c * (n ** power) for power, c in p.items())


def poly_str(p, unit_bytes=KiB, unit_name="KiB"):
    """Pretty-print a polynomial in n, trying to factor common shapes."""
    if not p:
        return f"0 {unit_name}"
    keys = sorted(p)
    if keys == [1]:                                 # c * n
        return f"{p[1]/unit_bytes:,.2f}*n {unit_name}"
    if keys == [0]:                                 # constant
        return f"{p[0]/unit_bytes:,.2f} {unit_name}"
    if keys == [0, 1] and p[0] == -p[1]:            # c*(n-1)
        return f"{p[1]/unit_bytes:,.2f}*(n-1) {unit_name}"
    if keys == [1, 2] and p[1] == -p[2]:            # c*n*(n-1)
        return f"{p[2]/unit_bytes:,.2f}*n*(n-1) {unit_name}"
    if keys == [0, 2] and p[0] == -p[2]:            # c*(n-1)*(n+1)
        return f"{p[2]/unit_bytes:,.2f}*(n-1)*(n+1) {unit_name}"
    terms = []
    for power in sorted(p, reverse=True):
        c = p[power] / unit_bytes
        var = {0: "", 1: "*n", 2: "*n^2"}[power]
        sign = " + " if c >= 0 and terms else (" - " if c < 0 and terms else
                                               ("-" if c < 0 else ""))
        terms.append(f"{sign}{abs(c):,.2f}{var}")
    return "".join(terms) + f" {unit_name}"


# ---------------------------------------------------------------------------
# Cost helper.  Pilvi decryption is non-interactive: the only message is each
# party's pd_k sent to a single combiner.  No broadcast / commit / open steps.
# ---------------------------------------------------------------------------

def to_single_combiner(data):
    """All n parties send `data` to one combiner: system = n * data, 1 round."""
    return dict(per_party={0: data},
                system   ={1: data},                    # n * data
                rounds   =1)


# ---------------------------------------------------------------------------
# Core cost computation.
# ---------------------------------------------------------------------------
def _cost(name):
    if name not in PARAMS:
        return None
    p = PARAMS[name]
    phi, log2q = PHI, p["log2q"]
    L = L_MSG
    n_lwe = p["n"]                                  # module rank (height of A)

    w   = ring_bytes(phi, log2q)                    # one ring element
    ct  = (n_lwe + L) * w                           # |ctxt| = (n+L) ring elements
    pd  = L * w                                     # |pd_k| = L ring elements
    sk  = L * n_lwe * w                             # master secret
    sk_share = {"const": sk, "log": 0}              # Shamir over rings -> = |sk|, N-indep

    # Phase 0: distribute ct  (encryptor -> n participating parties)
    P0 = dict(per_party={0: ct}, system={1: ct}, rounds=1)

    # Phase 1: collect partial decryptions.  NON-INTERACTIVE -- there are NO
    # commit / open rounds (unlike the interactive TKEMs).  Each of the n
    # participating parties sends its one pd_k to a single combiner: n * pd.
    step = to_single_combiner(pd)

    P1_per_party = step["per_party"]
    P1_system    = step["system"]
    P1_rounds    = step["rounds"]

    return dict(
        name=name, params=p,
        sizes=dict(sk=sk, sk_share=sk_share, pk_ek=None, ct=ct,
                   pd=pd, w=w, n_lwe=n_lwe, log2q=log2q,
                   t=p["t"], K=p["K"], Q=p["Q"]),
        P0=P0,
        P1=dict(step=step,
                total_per_party=P1_per_party,
                total_system=P1_system,
                total_rounds=P1_rounds),
        total_per_party=poly_add(P0["per_party"], P1_per_party),
        total_system   =poly_add(P0["system"],    P1_system),
        total_rounds   =P0["rounds"] + P1_rounds,
    )


def cost_pilvi(name):
    return _cost(name)


# ---------------------------------------------------------------------------
# Printer
# ---------------------------------------------------------------------------
def _fmt(b):
    if b is None:
        return "n/a"
    return f"{b/KiB:,.1f} KiB" if b >= KiB else f"{b:,.0f} B"


def sk_share_str(rep, unit_bytes=KiB, unit_name="KiB"):
    c, l = rep["const"], rep["log"]
    if l == 0:
        return f"{c/unit_bytes:,.2f} {unit_name}  (Shamir, N-independent)"
    return f"{c/unit_bytes:,.2f} + {l/unit_bytes:,.2f}*log2(N) {unit_name}"


def _print_step(label, step, n_concrete=None):
    pp = poly_str(step["per_party"])
    sy = poly_str(step["system"])
    r  = step["rounds"]
    print(f"        {label}")
    print(f"            per party : {pp}")
    line = f"            system    : {sy}    ({r} round{'s' if r!=1 else ''})"
    if n_concrete is not None:
        line += f"   |  at n={n_concrete}: {_fmt(poly_eval(step['system'], n_concrete))}"
    print(line)


def show(res):
    if res is None:
        print("     (no such parameter set)")
        return
    s = res["sizes"]
    t = s["t"]
    Qs = f"2^{int(round(__import__('math').log2(s['Q'])))}" if s["Q"] >= 2 else str(s["Q"])
    print(f"\n  -- {res['name']}  (t={t}, K={s['K']}, Q={Qs}, n_lwe={s['n_lwe']}, "
          f"log2 q={s['log2q']}) --")
    print(f"     regime = combiner (z_i sent to combiner, counted n*pd)  "
          f"[non-interactive: 1 send/party, no commit/open]")
    print(f"     sizes  : |sk|={_fmt(s['sk'])}   |sk_share|={sk_share_str(s['sk_share'])}")
    print(f"              |ctxt|={_fmt(s['ct'])}   |pd_k|={_fmt(s['pd'])}   "
          f"|w|={_fmt(s['w'])}")

    # decryption uses exactly t shares -> evaluate the polynomials at n = t
    nt = t
    _print_step("Phase 0  distribute ctxt :", res["P0"], n_concrete=nt)
    print(f"        Phase 1  collect partial decryptions (NON-INTERACTIVE):")
    _print_step("step    send pd_k -> reconstructor :", res["P1"]["step"], n_concrete=nt)
    print(f"            Phase 1 per-party total : {poly_str(res['P1']['total_per_party'])}")
    print(f"            Phase 1 system    total : {poly_str(res['P1']['total_system'])}"
          f"    ({res['P1']['total_rounds']} round"
          f"{'s' if res['P1']['total_rounds']!=1 else ''})")

    print(f"     END-TO-END per-party total : {poly_str(res['total_per_party'])}")
    print(f"     END-TO-END system    total : {poly_str(res['total_system'])}"
          f"    ({res['total_rounds']} rounds)")
    print(f"     END-TO-END system    total @ n=t={nt} : "
          f"{_fmt(poly_eval(res['total_system'], nt))}")


PILVI_NOTE = """\
Note on how |sk|, |sk_share|, |ctxt|, |pd_k| are computed (Pilvi, secpar=128, f=512):

   phi = euler_phi(512) = 256 ,  L = ceil(2*secpar/phi) = 1  (256-bit message)
   |w|  = phi * ceil(log2 q) / 8                         one ring element of R_q

   |ctxt|     = (n + L) * |w|       Fig. 5 Enc: ctxt = (c0 in R_q^n , c_l in R_q^L)
   |pd_k|     = L * |w|             Fig. 5 ParDec: pd_k = (s_{l,k}^T c0 + e)_{l in [L]}
                                    -> L ring elements  (the headline "small share")
   |sk|       = L * n * |w|         master secret r = (r_l)_{l in [L]}, r_l in R_q^n
   |sk_share| = L * n * |w| = |sk|  Shamir secret sharing over the subtractive set:
                                    each share s_{l,k} = (V R_l)_k is ONE row in R_q^n.
                                    No compression, and N-INDEPENDENT (cf. Paper A).

   Communication: NON-INTERACTIVE.  Each of the t participating parties sends ONE
   message pd_k to a single combiner -> system = n * pd_k.  No commit / no open /
   no inter-party rounds.  Decryption needs exactly t shares, so concrete system
   totals are reported at n = t.
"""


def report():
    print("\n" + "=" * 78)
    print("PILVI  -  Cini, Lai & Woo, Lattice Threshold PKE (2025-1691)")
    print("=" * 78)
    print(PILVI_NOTE)
    for name in PARAMS:
        show(cost_pilvi(name))


if __name__ == "__main__":
    report()