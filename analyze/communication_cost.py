"""
Threshold-decapsulation communication cost model for \\scheme (Moiragus).

Online-phase communication only; sk and pk sizes live in a separate script.
ct (encapsulating ciphertext) is included in Phase 0 (CCN -> n authorities).

Communication model:

  BROADCAST model           per party = (n-1)*data ; system = n*(n-1)*data ; 1 round
  TO-SINGLE-COMBINER model  per party =  data      ; system =  n*data      ; 1 round
  FROM-SINGLE-SOURCE model  per party =  data      ; system =  n*data      ; 1 round

ACTIVE VARIANT (SPDZ_{2^k}, q = 2^k, no modulus switch):
    step 4   open masked errors : broadcast ell polys over Z_{q/2}      (log2 q - 1 bits)
    step 4b  batch MAC check    : broadcast ONE coefficient over the
                                  lifted ring Z_{2^{k+ls}}              (log2 q + ls bits)
    final    partials to CCN    : to-CCN    ell polys over Z_{q}        (log2 q bits)
    rounds = 3   (+1 for Phase 0 distribute ct)

PASSIVE VARIANT:
    step    open masked errors : broadcast ell polys over Z_{mu}        (log2 mu bits)
    final   partials to CCN    : to-CCN    ell polys over Z_{q'}        (log2 q' bits)
    rounds = 2   (+1 for Phase 0 distribute ct)

PER-PARTY end-to-end is reported TWICE: excluding Phase 0 (server passively
receives ct from the CCN) and including Phase 0 (full per-server traffic).
SYSTEM end-to-end always includes Phase 0.

Output convention: BYTES first, then (KiB) highlighted in GREEN, e.g.
    8,800 B (8.59 KiB)   and   10,240.00*n - 2,048.00 B (10.00*n - 2.00 KiB)
"""
from math import ceil

KiB = 1024
GREEN = "\033[92m"
RESET = "\033[0m"


# ---------------------------------------------------------------------------
# Polynomial in n, stored as {power: coeff_in_bytes}.
# ---------------------------------------------------------------------------
def poly_add(*polys):
    out = {}
    for p in polys:
        for k, v in p.items():
            out[k] = out.get(k, 0) + v
    return {k: v for k, v in out.items() if v != 0}


def _poly_core(p, unit_bytes, fmt):
    """Render polynomial coefficients divided by unit_bytes, with `fmt` per coeff."""
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
    if keys == [0, 2] and p[0] == -p[2]:
        return f"{fmt(c(2))}*(n-1)*(n+1)"
    terms = []
    for power in sorted(p, reverse=True):
        cc = c(power)
        var = {0: "", 1: "*n", 2: "*n^2"}[power]
        sign = " + " if cc >= 0 and terms else (" - " if cc < 0 and terms else
                                                ("-" if cc < 0 else ""))
        terms.append(f"{sign}{fmt(abs(cc))}{var}")
    return "".join(terms)


def poly_str(p):
    """Bytes first, then (KiB) in green parentheses."""
    bytes_part = _poly_core(p, 1,   lambda x: f"{x:,.2f}")
    kib_part   = _poly_core(p, KiB, lambda x: f"{x:,.2f}")
    return f"{bytes_part} B    ({GREEN}{kib_part} KiB{RESET})"


# ---------------------------------------------------------------------------
# Cost helpers per opening regime.
# ---------------------------------------------------------------------------
def broadcast_step(data):
    return dict(per_party={1: data, 0: -data}, system={2: data, 1: -data}, rounds=1)


def to_single_combiner(data):
    return dict(per_party={0: data}, system={1: data}, rounds=1)


def from_single_source(data):
    return dict(per_party={0: data}, system={1: data}, rounds=1)


def _poly_bytes(d, log2_modulus):
    """d coefficients of size log2_modulus bits each, in bytes."""
    return d * ceil(log2_modulus) / 8.0


# ---------------------------------------------------------------------------
# Passive (semi-honest) variant
# ---------------------------------------------------------------------------
def cost_scheme_passive(log2_q, log2_qprime, log2_mu_prime, ell, d, ct_bytes):
    """
    Communication cost for Moiragus passive variant.

    Two-round release protocol:
    - P0: TEE distributes the encapsulating ciphertext to all KBS authorities.
    - P1, step 1: Authorities open masked errors among themselves (broadcast).
                  Each opened value is a polynomial mod μ', and there are ℓ of them.
    - P1, step 2: Authorities send their partial decryptions to the TEE
                  (single combiner). Each partial is a polynomial mod q', ℓ in total.
    """
    poly_mu = _poly_bytes(d, log2_mu_prime)  # element of Z_{μ'}^d
    poly_qprime = _poly_bytes(d, log2_qprime)  # element of Z_{q'}^d

    # P0: TEE -> all authorities, distributes ct_ak
    P0 = from_single_source(ct_bytes)

    # P1 step 1: authorities open ℓ masked-error polynomials mod μ' (all-to-all)
    step_open = broadcast_step(ell * poly_mu)

    # P1 step 2: each authority sends ℓ partial decryptions mod q' to TEE
    step_to_ccn = to_single_combiner(ell * poly_qprime)

    P1_per_party = poly_add(step_open["per_party"], step_to_ccn["per_party"])
    P1_system = poly_add(step_open["system"], step_to_ccn["system"])
    P1_rounds = step_open["rounds"] + step_to_ccn["rounds"]

    result =  dict(
        paper="Moiragus-passive",
        robust=False,
        sizes=dict(
            poly_mu=poly_mu,
            poly_qprime=poly_qprime,
            ell=ell,
            d=d,
            ct=ct_bytes,
            log2_q=log2_q,
            log2_qprime=log2_qprime,
            log2_mu=log2_mu_prime,
        ),
        P0=dict(
            steps=dict(distribute_ct=P0),
            per_party=P0["per_party"],
            system=P0["system"],
            rounds=P0["rounds"],
        ),
        P1=dict(
            steps=dict(
                step_open_masked_errors=step_open,
                step_send_partials_to_ccn=step_to_ccn,
            ),
            total_per_party=P1_per_party,
            total_system=P1_system,
            total_rounds=P1_rounds,
        ),
        total_per_party_excl_P0=P1_per_party,
        total_per_party_incl_P0=poly_add(P0["per_party"], P1_per_party),
        total_system=poly_add(P0["system"], P1_system),
        total_rounds=P0["rounds"] + P1_rounds,
    )

    show(result, "passive")


# ---------------------------------------------------------------------------
# Active (malicious-secure) variant  --  SPDZ_{2^k}, q = 2^k
# ---------------------------------------------------------------------------
def cost_scheme_active(log2_q, ell, d, ct_bytes, lambda_s):
    poly_open  = _poly_bytes(d, log2_q - 1 + lambda_s)            # ell polys over Z_{q/2}
    poly_mac   = _poly_bytes(1, log2_q + lambda_s)     # ONE coeff over lifted ring
    poly_final = _poly_bytes(d, log2_q + lambda_s)                # ell polys over Z_q

    P0             = from_single_source(ct_bytes)
    step_open      = broadcast_step(ell * poly_open)       # ell polys over q/2
    step_mac_check = broadcast_step(poly_mac)              # ONE coeff lifted
    step_to_ccn    = to_single_combiner(ell * poly_final)  # ell polys over q

    P1_per_party = poly_add(step_open["per_party"], step_mac_check["per_party"],
                            step_to_ccn["per_party"])
    P1_system    = poly_add(step_open["system"], step_mac_check["system"],
                            step_to_ccn["system"])
    P1_rounds    = step_open["rounds"] + step_mac_check["rounds"] + step_to_ccn["rounds"]

    result =  dict(
        paper="Moiragus-active", robust=True,
        sizes=dict(poly_open=poly_open, poly_mac=poly_mac, poly_final=poly_final,
                   ell=ell, d=d, ct=ct_bytes, log2_q=log2_q, lambda_s=lambda_s),
        P0=dict(steps=dict(distribute_ct=P0),
                per_party=P0["per_party"], system=P0["system"], rounds=P0["rounds"]),
        P1=dict(steps=dict(step_open_masked_errors=step_open,
                           step_batch_mac_check=step_mac_check,
                           step_send_partials_to_ccn=step_to_ccn),
                total_per_party=P1_per_party, total_system=P1_system,
                total_rounds=P1_rounds),
        total_per_party_excl_P0=P1_per_party,
        total_per_party_incl_P0=poly_add(P0["per_party"], P1_per_party),
        total_system           =poly_add(P0["system"],    P1_system),
        total_rounds           =P0["rounds"] + P1_rounds,
    )
    show(result, "active (SPDZ_{2^k})")


# ---------------------------------------------------------------------------
# Printer  (bytes first, KiB in green parentheses)
# ---------------------------------------------------------------------------
def _both(b):
    """Format a byte count as 'X B (Y KiB)' with KiB highlighted green."""
    b_str = f"{int(b):,} B" if b == int(b) else f"{b:,.0f} B"
    k_str = f"{b/KiB:,.2f} KiB"
    return f"{b_str}    ({GREEN}{k_str}{RESET})"


def _print_derivation(res):
    """Show input parameters and per-message size derivation."""
    s = res["sizes"]
    paper = res["paper"]
    print("     --- Inputs & size derivation ---")
    print(f"     ell={s['ell']}  d={s['d']}  |ct|={_both(s['ct'])}")

    if paper.endswith("active"):
        q  = s["log2_q"]
        ls = s["lambda_s"]
        print(f"     log2 q = {q}   lambda_s = {ls}")
        print(f"     |poly_open(q/2)| = d * (log2 q - 1) / 8 = {s['d']} * {q-1} / 8")
        print(f"                      = {_both(s['poly_open'])}    "
              f"[ell polys broadcast in step 4, modulus Z_{{q/2}}]")
        print(f"     |poly_mac(lift)| = 1 * (log2 q + lambda_s) / 8 = 1 * {q+ls} / 8")
        print(f"                      = {_both(s['poly_mac'])}    "
              f"[ONE coefficient over lifted ring Z_{{2^{{k+ls}}}} -- batch MAC check]")
        print(f"     |poly_final(q)|  = d * log2 q / 8 = {s['d']} * {q} / 8")
        print(f"                      = {_both(s['poly_final'])}    "
              f"[ell polys sent to CCN, modulus Z_q]")
        print(f"     step 4   broadcasts ell * poly_open = {s['ell']} * {int(s['poly_open']):,} B "
              f"= {_both(s['ell'] * s['poly_open'])}")
        print(f"     step 4b  broadcasts {_both(s['poly_mac'])}    (one coeff, not d)")
        print(f"     final    sends to CCN ell * poly_final = {s['ell']} * {int(s['poly_final']):,} B "
              f"= {_both(s['ell'] * s['poly_final'])}")
    else:  # passive
        print(f"     log2 q={s['log2_q']}   log2 q'={s['log2_qprime']}   log2 mu={s['log2_mu']}")
        print(f"     |poly_mu|     = d * log2 mu / 8     = {s['d']} * {s['log2_mu']} / 8")
        print(f"                   = {_both(s['poly_mu'])}    "
              f"[ell polys broadcast in step 4, modulus Z_mu]")
        print(f"     |poly_qprime| = d * log2 q' / 8     = {s['d']} * {s['log2_qprime']} / 8")
        print(f"                   = {_both(s['poly_qprime'])}    "
              f"[ell polys sent to CCN, modulus Z_q']")
        print(f"     step 4  broadcasts ell * poly_mu     = {s['ell']} * {int(s['poly_mu']):,} B "
              f"= {_both(s['ell'] * s['poly_mu'])}")
        print(f"     final   sends to CCN ell * poly_qprime = {s['ell']} * {int(s['poly_qprime']):,} B "
              f"= {_both(s['ell'] * s['poly_qprime'])}")


def _print_step(label, step):
    print(f"        {label}")
    print(f"            per party : {poly_str(step['per_party'])}")
    print(f"            system    : {poly_str(step['system'])}    "
          f"({step['rounds']} round{'s' if step['rounds'] != 1 else ''})")


def show(res, label):
    print(f"\n  ============ {label} ============")
    if res is None:
        print("     (no result)")
        return
    print(f"     regime=round-optimal (broadcast; partials to CCN)  robust={res['robust']}")
    _print_derivation(res)

    print(f"     --- Communication ---")
    print(f"        Phase 0  distribute ct :")
    for sk, step in res["P0"]["steps"].items():
        _print_step(sk + " :", step)

    print(f"        Phase 1  online communication :")
    step_labels = {
        "step_open_masked_errors":   "step open masked errors (q/2) :",
        "step_batch_mac_check":      "step batch MAC check (lifted) :",
        "step_send_partials_to_ccn": "step send partials to CCN (q) :",
    }
    for sk, lbl in step_labels.items():
        if sk in res["P1"]["steps"]:
            _print_step(lbl, res["P1"]["steps"][sk])
    print(f"            Phase 1 per-party total : {poly_str(res['P1']['total_per_party'])}")
    print(f"            Phase 1 system    total : {poly_str(res['P1']['total_system'])}"
          f"    ({res['P1']['total_rounds']} rounds)")

    print(f"     END-TO-END per-party total (excl. Phase 0) : "
          f"{poly_str(res['total_per_party_excl_P0'])}")
    print(f"     END-TO-END per-party total (incl. Phase 0) : "
          f"{poly_str(res['total_per_party_incl_P0'])}")
    print(f"     END-TO-END system    total                 : "
          f"{poly_str(res['total_system'])}    ({res['total_rounds']} rounds)")

