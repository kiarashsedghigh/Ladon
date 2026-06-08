"""
Threshold-decryption communication cost model for Micciancio-Suhl.

Paper : Micciancio & Suhl, "Simulation-Secure Threshold PKE from LWE with
        Polynomial Modulus" (2023-1728).

================================================================================
STRUCTURE
================================================================================
Like Pilvi, this scheme is NON-INTERACTIVE: each server locally computes one
partial decryption and the shares are simply ADDED UP and rounded.  There are NO
commit / open rounds (unlike the Lapiha-Prest interactive TKEMs).  So Phase 1
is a single message per party to one reconstructor (combiner), counted as
   system = n * |pd|.

Differences from Pilvi worth flagging:
  * It is T-out-of-T (additive sharing  s = sum_i s_i), NOT t-out-of-K.  All T
    parties must participate; the symbolic variable n here = T = #parties.
  * It is PLAIN LWE (not ring): a "coefficient" is one Z_q integer of
    ceil(log2 q) bits, i.e. no ring degree multiplier.
  * Additive T-out-of-T shares are full uniform vectors -> |sk_share| = |sk|,
    N-independent.

================================================================================
COMMUNICATION MODEL (n = T = #participating parties)
================================================================================
  Phase 0   distribute ctxt    encryptor -> n parties   ; system = n * |ctxt|  (1 round)
  Phase 1   collect shares     each party -> combiner   ; system = n * |pd|    (1 round)
                               (shares summed by the combiner; non-interactive)

PER-PARTY end-to-end is reported TWICE: excluding Phase 0 (party only receives
ctxt) and including Phase 0 (full per-server traffic). SYSTEM always includes
Phase 0.  Output convention: BYTES first, KiB in GREEN parentheses.
"""
from math import ceil, log2

KiB = 1024
GREEN = "\033[92m"
RESET = "\033[0m"


# ---------------------------------------------------------------------------
# Parameters.  Section 6 Frodo-640-like instantiation (128-bit security).
# ---------------------------------------------------------------------------
SECPAR  = 128
LWE_N   = 640
LOG2Q   = 17                 # ceil(log2 65537);  16-bit packing possible (see note)
BITS_PER_CT = 2              # plaintext modulus 4 -> 2 bits per scalar ciphertext
MSG_BITS = 256
NUM_CT  = MSG_BITS // BITS_PER_CT        # = 128 independent ciphertexts

# deployment sizes T (= n) to report; paper supports up to T = 8263
T_VALUES = [2, 8, 32, 256, 8263]


def elem_bytes(log2q):
    """One Z_q integer: ceil(log2 q) bits / 8  (plain LWE -- no ring degree)."""
    return ceil(log2q) / 8.0


# ---------------------------------------------------------------------------
# Polynomial in n.
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
    keys = sorted(p)
    def c(power):
        return p[power] / unit_bytes
    if keys == [1]:
        return f"{fmt(c(1))}*n"
    if keys == [0]:
        return f"{fmt(c(0))}"
    if keys == [0, 1] and p[0] == -p[1]:
        return f"{fmt(c(1))}*(n-1)"
    if keys == [1, 2] and p[1] == -p[2]:
        return f"{fmt(c(2))}*n*(n-1)"
    terms = []
    for power in sorted(p, reverse=True):
        cc = c(power)
        var = {0: "", 1: "*n", 2: "*n^2"}[power]
        sign = " + " if cc >= 0 and terms else (" - " if cc < 0 and terms else
                                                ("-" if cc < 0 else ""))
        terms.append(f"{sign}{fmt(abs(cc))}{var}")
    return "".join(terms)


def poly_str(p):
    """Bytes first, then (KiB) in GREEN parentheses."""
    bytes_part = _poly_core(p, 1,   lambda x: f"{x:,.3f}")
    kib_part   = _poly_core(p, KiB, lambda x: f"{x:,.3f}")
    return f"{bytes_part} B    ({GREEN}{kib_part} KiB{RESET})"


# ---------------------------------------------------------------------------
# Cost helper.  Non-interactive: only message is pd_k to a single combiner.
# ---------------------------------------------------------------------------
def to_single_combiner(data):
    return dict(per_party={0: data},
                system   ={1: data},                      # n * data
                rounds   =1)


# ---------------------------------------------------------------------------
# Core cost computation.
# ---------------------------------------------------------------------------
def _cost(log2q=LOG2Q, n_lwe=LWE_N, nc=NUM_CT):
    e   = elem_bytes(log2q)
    w   = e
    ct1 = (n_lwe + 1) * e                        # one scalar ciphertext
    ct  = nc * ct1                               # full 256-bit message
    pd  = nc * e                                 # |pd_k|
    sk  = n_lwe * e                              # master secret
    sk_share = {"const": sk, "log": 0}           # additive T-of-T, N-indep
    pk  = (2 * SECPAR / 8.0) + n_lwe * e + 4

    # Phase 0: distribute ct  (encryptor -> n participating parties)
    P0 = dict(per_party={0: ct}, system={1: ct}, rounds=1)

    # Phase 1: collect partial decryptions (NON-INTERACTIVE).
    step = to_single_combiner(pd)

    return dict(
        sizes=dict(sk=sk, sk_share=sk_share, pk=pk, ct=ct, ct1=ct1,
                   pd=pd, w=w, n_lwe=n_lwe, log2q=log2q, nc=nc),
        P0=P0,
        P1=dict(step=step,
                total_per_party=step["per_party"],
                total_system=step["system"],
                total_rounds=step["rounds"]),
        total_per_party_excl_P0=step["per_party"],
        total_per_party_incl_P0=poly_add(P0["per_party"], step["per_party"]),
        total_system           =poly_add(P0["system"],    step["system"]),
        total_rounds           =P0["rounds"] + step["rounds"],
    )


def cost_micciancio_suhl():
    return _cost()


# ---------------------------------------------------------------------------
# Printer (bytes first, KiB in green parentheses)
# ---------------------------------------------------------------------------
def _both(b):
    if b is None:
        return "n/a"
    if b == int(b) and b < 10_000:
        b_str = f"{int(b):,} B"
    else:
        b_str = f"{b:,.0f} B"
    if b >= KiB * KiB:
        k_str = f"{b/(KiB*KiB):,.2f} MiB"
    else:
        k_str = f"{b/KiB:,.2f} KiB"
    return f"{b_str}    ({GREEN}{k_str}{RESET})"


def sk_share_str(rep):
    c, l = rep["const"], rep["log"]
    if l == 0:
        return f"{_both(c)}    (additive T-of-T, N-independent)"
    return (f"{c:,.0f} + {l:,.0f}*log2(N) B    "
            f"({GREEN}{c/KiB:,.2f} + {l/KiB:,.2f}*log2(N) KiB{RESET})")


def _print_step(label, step, n_concrete=None):
    print(f"        {label}")
    print(f"            per party : {poly_str(step['per_party'])}")
    line = (f"            system    : {poly_str(step['system'])}    "
            f"({step['rounds']} round{'s' if step['rounds']!=1 else ''})")
    if n_concrete is not None:
        line += f"   |  at n={n_concrete}: {_both(poly_eval(step['system'], n_concrete))}"
    print(line)


def show(res):
    s = res["sizes"]
    print(f"\n  -- Micciancio-Suhl  (Frodo-640-like, secpar=128, plain LWE) --")
    print(f"     n_lwe={s['n_lwe']}, q=2^16+1 (log2 q={s['log2q']}), "
          f"{BITS_PER_CT} bits/ciphertext -> nc={s['nc']} ciphertexts for a "
          f"{s['nc']*BITS_PER_CT}-bit message")
    print(f"     regime = combiner (pd_k summed at one combiner, counted n*pd)  "
          f"[non-interactive: 1 send/party, no commit/open]")
    print(f"     --- Sizes ---")
    print(f"     |sk|       = {_both(s['sk'])}")
    print(f"     |sk_share| = {sk_share_str(s['sk_share'])}")
    print(f"     |pk|       = {_both(s['pk'])}")
    print(f"     |ctxt(1 scalar)|  = {_both(s['ct1'])}")
    print(f"     |ctxt(256-bit)|   = {_both(s['ct'])}    (= nc * (n+1) * elem)")
    print(f"     |pd_k|     = {_both(s['pd'])}    (= nc * elem)")
    print(f"     |elem|     = {_both(s['w'])}    (one Z_q integer)")

    print(f"     --- Communication ---")
    _print_step("Phase 0  distribute ctxt :", res["P0"])
    print(f"        Phase 1  collect partial decryptions (NON-INTERACTIVE):")
    _print_step("step    send pd_k -> combiner :", res["P1"]["step"])

    print(f"     END-TO-END per-party total (excl. Phase 0) : "
          f"{poly_str(res['total_per_party_excl_P0'])}")
    print(f"     END-TO-END per-party total (incl. Phase 0) : "
          f"{poly_str(res['total_per_party_incl_P0'])}")
    print(f"     END-TO-END system    total                 : "
          f"{poly_str(res['total_system'])}    ({res['total_rounds']} rounds)")

    # T-out-of-T: evaluate at several deployment sizes T (= n)
    print(f"     --- End-to-end at concrete n = T parties ---")
    print(f"     {'T':<8} {'per-party (excl. P0)':<22} {'per-party (incl. P0)':<22} "
          f"{'system':<22}")
    for T in T_VALUES:
        tag = "  (max supported)" if T == 8263 else ""
        pp_e = poly_eval(res['total_per_party_excl_P0'], T)
        pp_i = poly_eval(res['total_per_party_incl_P0'], T)
        sy   = poly_eval(res['total_system'], T)
        # plain (no color) for the table to keep columns aligned
        def _plain(b):
            if b >= KiB*KiB: return f"{b/(KiB*KiB):,.2f} MiB"
            return f"{b/KiB:,.2f} KiB" if b >= KiB else f"{b:,.0f} B"
        print(f"     T={T:<6} {_plain(pp_e):<22} {_plain(pp_i):<22} "
              f"{_plain(sy):<22}{tag}")


MS_NOTE = """\
Note on how |sk|, |sk_share|, |pk|, |ctxt|, |pd_k| are computed
(Micciancio-Suhl, Section 6 Frodo-640-like params, scaled to a 256-bit message):

   plain LWE  ->  one element of Z_q = ceil(log2 q)/8 bytes  (no ring degree)
   n = 640 , q = 65537 (=2^16+1, prime) so ceil(log2 q) = 17 bits = 2.125 B/elem
       (q is barely above 2^16; a 16-bit packing is possible if the single value
        65536 is special-cased -- that would shave ~1/17 off every size below.)

   EACH CIPHERTEXT ENCRYPTS ONE SCALAR (no matrix variant / no a-part reuse).
   Plaintext modulus 4 carries 2 bits per scalar, so a 256-bit message is sent as
        nc = 256 / 2 = 128  INDEPENDENT ciphertexts, each with its OWN a-part.

   1 scalar ciphertext = (r^T A + f in Z_q^n , scalar in Z_q)        = (n+1) elem
   |ctxt| (256-bit)    = nc * (n+1) * elem        nc independent ciphertexts
   |pd_k|              = nc * elem                one scalar <a,s_k>+e~ per ctxt
   |sk|                = n * elem                 ONE secret s in Z_q^n decrypts
                                                  ALL nc ciphertexts (no per-ct keys)
   |sk_share|          = n * elem = |sk|          additive T-of-T share s = sum_i s_i;
                                                  full uniform vector, N-INDEPENDENT.
   |pk|                = seed_A (2*secpar bits) + b=As+e_pk (n elem) + norm c (1 int)

   Communication: NON-INTERACTIVE and T-OUT-OF-T.  All T parties participate;
   each sends ONE pd_k to a single combiner which sums them -> system = T * |pd_k|.
   No commit / open / inter-party rounds.  n (= T) is the #participating parties.
"""


def report():
    print("\n" + "=" * 78)
    print("MICCIANCIO-SUHL  -  Simulation-Secure Threshold PKE, poly modulus (2023-1728)")
    print("=" * 78)
    print(MS_NOTE)
    show(cost_micciancio_suhl())


if __name__ == "__main__":
    report()