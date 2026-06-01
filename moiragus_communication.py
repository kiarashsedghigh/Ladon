"""
Threshold-decapsulation communication cost model for \scheme (Moiragus).

Online-phase communication only; sk and pk sizes live in a separate script.
ct (encapsulating ciphertext) is included in Phase 0 (CCN -> n authorities).

Communication model (matches the Lapiha/Boudgoust comparison script):

  BROADCAST model           a party broadcasts to the OTHER (n-1) parties:
                              per party    = (n-1) * data
                              system total =  n * (n-1) * data
                              rounds       = 1

  TO-SINGLE-COMBINER model  every party sends `data` to one designated party
                            (here: the confidential computing node CCN):
                              per party    = data
                              system total = n * data
                              rounds       = 1

  FROM-SINGLE-SOURCE model  one source (here: CCN) sends `data` to each of
                            the n authorities:
                              per party    = data
                              system total = n * data
                              rounds       = 1

Communication for \scheme:

  Phase 0  distribute ct          : CCN -> n authorities, system = n * |ct|

  PASSIVE VARIANT (\release.\partialdec, no MACs, modulus switch from q to q'):
    step 4  (open masked errors)  : broadcast  ell polys over Z_{mu'}
    final   (send partials to CCN): to-CCN     ell polys over Z_{q'}
    rounds  = 2

  ACTIVE VARIANT (SPDZ_{2^k}, q = 2^k, no modulus switch):
    shares live in lifted ring Z_{2^{k+lambda_s}}; every opened poly is sized
    over the lifted ring.
    step 4  (open masked errors)  : broadcast  ell polys over Z_{2^{k+lambda_s}}
    step 4b (batch MAC check)     : broadcast  one poly over Z_{2^{k+lambda_s}}
    final   (send partials to CCN): to-CCN     ell polys over Z_{2^{k+lambda_s}}
    rounds  = 3
"""
from math import ceil

KiB = 1024


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
    """Pretty-print a polynomial in n."""
    if not p:
        return f"0 {unit_name}"
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
    """Every party broadcasts `data` to the (n-1) others; 1 round."""
    return dict(
        per_party={1: data, 0: -data},     # (n-1) * data
        system   ={2: data, 1: -data},     # n * (n-1) * data
        rounds   =1,
    )


def to_single_combiner(data):
    """All n parties send `data` to one combiner (here: CCN); 1 round."""
    return dict(
        per_party={0: data},                # data
        system   ={1: data},                # n * data
        rounds   =1,
    )


def from_single_source(data):
    """One source (here: CCN) sends `data` to each of the n authorities; 1 round.
    From the authorities' perspective each receives `data`."""
    return dict(
        per_party={0: data},                # data (received)
        system   ={1: data},                # n * data
        rounds   =1,
    )


def _poly_bytes(d, log2_modulus):
    """One polynomial of degree d with coefficients of size log2_modulus bits."""
    return d * ceil(log2_modulus) / 8.0


# ---------------------------------------------------------------------------
# Passive (semi-honest) variant
# ---------------------------------------------------------------------------
def cost_moiragus_passive(log2_q, log2_qprime, log2_mu, ell, d, ct_bytes):
    """
    End-to-end communication for the passive variant of \scheme
    (Phase 0 distribution of ct + Phase 1 online).

    Parameters:
        log2_q       : bit size of the working modulus q (kept for symmetry)
        log2_qprime  : bit size of q' (post-modulus-switch modulus)
        log2_mu      : bit size of mu' = q'/2 (modulus the opened values live in)
        ell          : number of parallel masking copies
        d            : ring dimension
        ct_bytes     : size in bytes of the encapsulating ciphertext ct
    """
    poly_mu     = _poly_bytes(d, log2_mu)
    poly_qprime = _poly_bytes(d, log2_qprime)

    # Phase 0: CCN distributes ct to each of n authorities.
    P0 = from_single_source(ct_bytes)

    # Phase 1.
    step_open   = broadcast_step(ell * poly_mu)
    step_to_ccn = to_single_combiner(ell * poly_qprime)

    P1_per_party = poly_add(step_open["per_party"], step_to_ccn["per_party"])
    P1_system    = poly_add(step_open["system"],    step_to_ccn["system"])
    P1_rounds    = step_open["rounds"] + step_to_ccn["rounds"]

    return dict(
        paper="Moiragus-passive", robust=False,
        sizes=dict(poly_mu=poly_mu, poly_qprime=poly_qprime,
                   ell=ell, d=d, ct=ct_bytes,
                   log2_qprime=log2_qprime, log2_mu=log2_mu),
        P0=dict(steps=dict(distribute_ct=P0),
                per_party=P0["per_party"], system=P0["system"],
                rounds=P0["rounds"]),
        P1=dict(steps=dict(step_open_masked_errors=step_open,
                           step_send_partials_to_ccn=step_to_ccn),
                total_per_party=P1_per_party,
                total_system=P1_system,
                total_rounds=P1_rounds),
        total_per_party=poly_add(P0["per_party"], P1_per_party),
        total_system   =poly_add(P0["system"],    P1_system),
        total_rounds   =P0["rounds"] + P1_rounds,
    )


# ---------------------------------------------------------------------------
# Active (malicious-secure) variant
# ---------------------------------------------------------------------------
def cost_moiragus_active(log2_q, ell, d, ct_bytes, lambda_s=40):
    """
    End-to-end communication for the active variant of \scheme (SPDZ_{2^k}).

    Parameters:
        log2_q   : bit size of working modulus q = 2^k (no modulus switch)
        ell      : number of parallel masking copies
        d        : ring dimension
        ct_bytes : size in bytes of the encapsulating ciphertext ct
        lambda_s : SPDZ_{2^k} statistical-security parameter (default 40)
    """
    # Every share lives in the lifted ring Z_{2^{k + lambda_s}}.
    poly_lifted = _poly_bytes(d, log2_q + lambda_s)

    # Phase 0.
    P0 = from_single_source(ct_bytes)

    # Phase 1.
    step_open      = broadcast_step(ell * poly_lifted)
    step_mac_check = broadcast_step(poly_lifted)
    step_to_ccn    = to_single_combiner(ell * poly_lifted)

    P1_per_party = poly_add(step_open["per_party"],
                            step_mac_check["per_party"],
                            step_to_ccn["per_party"])
    P1_system    = poly_add(step_open["system"],
                            step_mac_check["system"],
                            step_to_ccn["system"])
    P1_rounds    = (step_open["rounds"] +
                    step_mac_check["rounds"] +
                    step_to_ccn["rounds"])

    return dict(
        paper="Moiragus-active", robust=True,
        sizes=dict(poly_lifted=poly_lifted, ell=ell, d=d, ct=ct_bytes,
                   log2_q=log2_q, lambda_s=lambda_s),
        P0=dict(steps=dict(distribute_ct=P0),
                per_party=P0["per_party"], system=P0["system"],
                rounds=P0["rounds"]),
        P1=dict(steps=dict(step_open_masked_errors=step_open,
                           step_batch_mac_check=step_mac_check,
                           step_send_partials_to_ccn=step_to_ccn),
                total_per_party=P1_per_party,
                total_system=P1_system,
                total_rounds=P1_rounds),
        total_per_party=poly_add(P0["per_party"], P1_per_party),
        total_system   =poly_add(P0["system"],    P1_system),
        total_rounds   =P0["rounds"] + P1_rounds,
    )


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
    print(f"            system    : {sy}    ({r} round{'s' if r != 1 else ''})")


def show(res, label):
    print(f"\n  -- {label} --")
    if res is None:
        print("     (no result)")
        return
    s = res["sizes"]
    print(f"     regime=round-optimal (broadcast; partials to CCN)  robust={res['robust']}")
    print(f"     sizes  : {s}")

    # Phase 0
    print(f"        Phase 0  distribute ct :")
    for sk, step in res["P0"]["steps"].items():
        _print_step(sk + " :", step)

    # Phase 1
    print(f"        Phase 1  online communication :")
    step_labels = {
        "step_open_masked_errors":     "step open masked errors    :",
        "step_batch_mac_check":        "step batch MAC check       :",
        "step_send_partials_to_ccn":   "step send partials to CCN  :",
    }
    for sk, lbl in step_labels.items():
        if sk in res["P1"]["steps"]:
            _print_step(lbl, res["P1"]["steps"][sk])
    print(f"            Phase 1 per-party total : {poly_str(res['P1']['total_per_party'])}")
    print(f"            Phase 1 system    total : {poly_str(res['P1']['total_system'])}"
          f"    ({res['P1']['total_rounds']} rounds)")

    print(f"     END-TO-END per-party total : {poly_str(res['total_per_party'])}")
    print(f"     END-TO-END system    total : {poly_str(res['total_system'])}"
          f"    ({res['total_rounds']} rounds)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 78)
    print("MOIRAGUS (this work)  -  end-to-end communication (Phase 0 + Phase 1)")
    print("=" * 78)

    # Passive variant: pass your concrete numbers below.
    moir_passive = cost_moiragus_passive(
        log2_q      = 12,         # placeholder: e.g. ML-KEM prime q = 3329 -> 12 bits
        log2_qprime = 11,         # placeholder: q' = 2^11 = 2048
        log2_mu     = 10,         # mu' = q'/2 = 2^10
        ell         = 8,          # placeholder
        d           = 256,        # ring dimension
        ct_bytes    = 1088,       # placeholder: e.g. ML-KEM-768 ciphertext
    )
    show(moir_passive, "passive (semi-honest)")

    # Active variant: pass your concrete numbers below.
    moir_active = cost_moiragus_active(
        log2_q   = 32,            # placeholder: q = 2^32
        ell      = 8,             # placeholder
        d        = 256,
        ct_bytes = 1088,          # placeholder
        lambda_s = 40,
    )
    show(moir_active, "active (SPDZ_{2^k})")