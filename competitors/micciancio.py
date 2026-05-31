"""
Threshold-decryption communication cost model for Micciancio-Suhl.

Paper : Micciancio & Suhl, "Simulation-Secure Threshold PKE from LWE with
        Polynomial Modulus" (2023-1728).

================================================================================
STRUCTURE (and how it relates to the other two scripts)
================================================================================
Like Pilvi, this scheme is NON-INTERACTIVE: each server locally computes one
partial decryption and the shares are simply ADDED UP and rounded.  There are NO
commit / open rounds (unlike the Lapiha-Prest interactive TKEMs in sasha_prest).
So Phase 1 is a single message per party to one reconstructor (combiner),
counted -- as agreed -- as system = n * |pd|.

Differences from Pilvi worth flagging:
  * It is T-out-of-T (additive sharing  s = sum_i s_i), NOT t-out-of-K.  So ALL
    T parties must participate; the symbolic variable n here = T = #parties.
  * It is PLAIN LWE (not ring): a "coefficient" is one Z_q integer of
    ceil(log2 q) bits, i.e. the per-element size is ceil(log2 q)/8 bytes with no
    ring degree phi multiplier.
  * Additive T-out-of-T shares are full uniform vectors, so |sk_share| = |sk|,
    N-independent (cf. the additive/Shamir N-independent cases elsewhere).

================================================================================
EXTENDING THE 2-BIT EXAMPLE TO 256-BIT MESSAGES
================================================================================
Section 6 gives a Frodo-640-like instantiation for a SINGLE scalar 2-bit message
(plaintext modulus 4).  The paper notes a matrix variant is "straightforward but
left for future work".  To match Pilvi's 256-bit message we do exactly that
matrix/reused-randomness extension:

    pack L = 256 / 2 = 128 two-bit slots into ONE ciphertext that REUSES a single
    a-part (r^T A + f in Z_q^n), exactly like Frodo's C1 and like Pilvi's c0=Ax.

    ctxt   = ( a-part in Z_q^n  [shared] , (b-slots) in Z_q^L )   -> (n + L) elems
    pd_k   = ( <a, s_{k,j}> + e~ )_{j in [L]}  in Z_q^L           -> L elems
    sk     = (s_1, ..., s_L) in (Z_q^n)^L                         -> L*n elems
    sk_k   = additive share, same shape                          -> L*n elems = |sk|

This is the SAME shape as Pilvi (just Z_q scalars in place of ring elements),
which is what makes the two directly comparable.

================================================================================
COMMUNICATION MODEL (n = T = #participating parties)
================================================================================
  Phase 0   distribute ctxt    encryptor -> n parties ; system = n * |ctxt|   (1 round)
  Phase 1   collect shares      each party -> ONE combiner ; system = n * |pd| (1 round)
                                (shares are summed by the combiner; non-interactive)
"""
from math import ceil, log2

KiB = 1024


# ---------------------------------------------------------------------------
# Parameters.  Section 6 Frodo-640-like instantiation (128-bit security).
#   n     = 640         (LWE secret dimension, matches Frodo-640)
#   q     = 65537       (= 2^16 + 1, prime;  ceil(log2 q) = 17 bits)
#   p     = 4           plaintext modulus -> 2 bits per scalar ciphertext
#   nc    = 256 / 2 = 128  INDEPENDENT scalar ciphertexts for a 256-bit message
# Each ciphertext encrypts ONE scalar (no matrix / no a-part reuse), so a
# 256-bit message is sent as nc separate ciphertexts, each with its OWN a-part.
# The scheme is T-out-of-T; we tabulate several deployment sizes T.
# ---------------------------------------------------------------------------
SECPAR  = 128
LWE_N   = 640
LOG2Q   = 17            # ceil(log2 65537);  16-bit packing possible (see note)
BITS_PER_CT = 2         # plaintext modulus 4 -> 2 bits per scalar ciphertext
MSG_BITS = 256
NUM_CT  = MSG_BITS // BITS_PER_CT        # = 128 independent ciphertexts

# deployment sizes T (= n) to report; paper supports up to T = 8263
T_VALUES = [2, 8, 32, 256, 8263]


def elem_bytes(log2q):
    """One Z_q integer: ceil(log2 q) bits / 8  (plain LWE -- no ring degree)."""
    return ceil(log2q) / 8.0


# ---------------------------------------------------------------------------
# Polynomial in n, stored as {power: coeff_in_bytes}.   (same machinery)
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
    if not p:
        return f"0 {unit_name}"
    keys = sorted(p)
    if keys == [1]:
        return f"{p[1]/unit_bytes:,.3f}*n {unit_name}"
    if keys == [0]:
        return f"{p[0]/unit_bytes:,.3f} {unit_name}"
    if keys == [0, 1] and p[0] == -p[1]:
        return f"{p[1]/unit_bytes:,.3f}*(n-1) {unit_name}"
    if keys == [1, 2] and p[1] == -p[2]:
        return f"{p[2]/unit_bytes:,.3f}*n*(n-1) {unit_name}"
    terms = []
    for power in sorted(p, reverse=True):
        c = p[power] / unit_bytes
        var = {0: "", 1: "*n", 2: "*n^2"}[power]
        sign = " + " if c >= 0 and terms else (" - " if c < 0 and terms else
                                               ("-" if c < 0 else ""))
        terms.append(f"{sign}{abs(c):,.3f}{var}")
    return "".join(terms) + f" {unit_name}"


# ---------------------------------------------------------------------------
# Cost helper.  Non-interactive: only message is pd_k to a single combiner.
# ---------------------------------------------------------------------------
def to_single_combiner(data):
    """All n parties send `data` to one combiner: system = n * data, 1 round."""
    return dict(per_party={0: data},
                system   ={1: data},                      # n * data
                rounds   =1)


# ---------------------------------------------------------------------------
# Core cost computation.
# ---------------------------------------------------------------------------
def _cost(log2q=LOG2Q, n_lwe=LWE_N, nc=NUM_CT):
    e   = elem_bytes(log2q)                      # one Z_q integer
    w   = e                                      # "one element" (for printing)
    # Each scalar ciphertext = (a-part in Z_q^n , 1 scalar in Z_q) = (n+1) elems.
    # A 256-bit message = nc INDEPENDENT such ciphertexts, each with its own a-part.
    ct1 = (n_lwe + 1) * e                        # one scalar ciphertext
    ct  = nc * ct1                               # full 256-bit message
    # Partial decryption: one scalar <a, s_k> + e~ per ciphertext -> nc scalars.
    pd  = nc * e                                 # |pd_k|
    # ONE secret s in Z_q^n decrypts ALL nc ciphertexts (no per-slot keys).
    sk  = n_lwe * e                              # master secret
    sk_share = {"const": sk, "log": 0}           # additive T-of-T share -> = |sk|, N-indep
    # public key: seed for A (2k bits) + b = As+e_pk (n elems) + norm c (1 int)
    pk  = (2 * SECPAR / 8.0) + n_lwe * e + 4

    # Phase 0: distribute ct  (encryptor -> n participating parties)
    P0 = dict(per_party={0: ct}, system={1: ct}, rounds=1)

    # Phase 1: collect partial decryptions.  NON-INTERACTIVE -- shares are summed
    # by one combiner; each of the n=T parties sends its pd_k once: n * pd.
    step = to_single_combiner(pd)

    return dict(
        sizes=dict(sk=sk, sk_share=sk_share, pk=pk, ct=ct, ct1=ct1,
                   pd=pd, w=w, n_lwe=n_lwe, log2q=log2q, nc=nc),
        P0=P0,
        P1=dict(step=step,
                total_per_party=step["per_party"],
                total_system=step["system"],
                total_rounds=step["rounds"]),
        total_per_party=poly_add(P0["per_party"], step["per_party"]),
        total_system   =poly_add(P0["system"],    step["system"]),
        total_rounds   =P0["rounds"] + step["rounds"],
    )


def cost_micciancio_suhl():
    return _cost()


# ---------------------------------------------------------------------------
# Printer
# ---------------------------------------------------------------------------
def _fmt(b):
    if b is None:
        return "n/a"
    if b >= KiB * KiB:
        return f"{b/(KiB*KiB):,.2f} MiB"
    return f"{b/KiB:,.2f} KiB" if b >= KiB else f"{b:,.0f} B"


def sk_share_str(rep, unit_bytes=KiB, unit_name="KiB"):
    c, l = rep["const"], rep["log"]
    if l == 0:
        return f"{c/unit_bytes:,.2f} {unit_name}  (additive T-of-T, N-independent)"
    return f"{c/unit_bytes:,.2f} + {l/unit_bytes:,.2f}*log2(N) {unit_name}"


def _print_step(label, step, n_concrete=None):
    print(f"        {label}")
    print(f"            per party : {poly_str(step['per_party'])}")
    line = (f"            system    : {poly_str(step['system'])}    "
            f"({step['rounds']} round{'s' if step['rounds']!=1 else ''})")
    if n_concrete is not None:
        line += f"   |  at n={n_concrete}: {_fmt(poly_eval(step['system'], n_concrete))}"
    print(line)


def show(res):
    s = res["sizes"]
    print(f"\n  -- Micciancio-Suhl  (Frodo-640-like, secpar=128, plain LWE) --")
    print(f"     n_lwe={s['n_lwe']}, q=2^16+1 (log2 q={s['log2q']}), "
          f"{BITS_PER_CT} bits/ciphertext -> nc={s['nc']} ciphertexts for a "
          f"{s['nc']*BITS_PER_CT}-bit message")
    print(f"     regime = combiner (pd_k summed at one combiner, counted n*pd)  "
          f"[non-interactive: 1 send/party, no commit/open]")
    print(f"     sizes  : |sk|={_fmt(s['sk'])}   |sk_share|={sk_share_str(s['sk_share'])}")
    print(f"              |pk|={_fmt(s['pk'])}   |ctxt(1 scalar)|={_fmt(s['ct1'])}   "
          f"|ctxt(256-bit)|={_fmt(s['ct'])}   |pd_k|={_fmt(s['pd'])}   |elem|={_fmt(s['w'])}")

    _print_step("Phase 0  distribute ctxt :", res["P0"])
    print(f"        Phase 1  collect partial decryptions (NON-INTERACTIVE):")
    _print_step("step    send pd_k -> combiner :", res["P1"]["step"])

    print(f"     END-TO-END per-party total : {poly_str(res['total_per_party'])}")
    print(f"     END-TO-END system    total : {poly_str(res['total_system'])}"
          f"    ({res['total_rounds']} rounds)")

    # T-out-of-T: evaluate the system total at several deployment sizes T (= n)
    print(f"     END-TO-END system total at n = T parties:")
    for T in T_VALUES:
        tag = "  (max supported)" if T == 8263 else ""
        print(f"          T={T:<5}: {_fmt(poly_eval(res['total_system'], T))}"
              f"   (per-party {_fmt(poly_eval(res['total_per_party'], T))}){tag}")


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