# jax_topk_bench.py
import time
from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jax import lax

# ----------------------------
# Methods under test
# ----------------------------

def topk_concat_full(a, b, k: int):
    v_all = jnp.concatenate([a, b], axis=0)
    vals, _ = lax.top_k(v_all, k)  # k must be static
    return vals

def topk_concat_2k(a, b, k: int):
    ka = min(int(k), a.size)  # use Python min, not jnp.minimum
    kb = min(int(k), b.size)
    va, _ = lax.top_k(a, ka)
    vb, _ = lax.top_k(b, kb)
    v_all = jnp.concatenate([va, vb], axis=0)
    vals, _ = lax.top_k(v_all, k)
    return vals

def _dtype_min(x):
    if jnp.issubdtype(x.dtype, jnp.integer):
        return jnp.iinfo(x.dtype).min
    else:
        return jnp.finfo(x.dtype).min

def topk_streaming(a, b, k: int):
    """Streaming merge of two top-k lists; returns only values."""
    k = int(k)
    ka = min(k, a.size)
    kb = min(k, b.size)

    va, _ = lax.top_k(a, ka)  # sorted desc
    vb, _ = lax.top_k(b, kb)  # sorted desc

    K = min(k, ka + kb)
    neg_inf = _dtype_min(va)

    def body(carry):
        i, j, t, out = carry
        a_done = i >= ka
        b_done = j >= kb
        ai = jnp.where(a_done, neg_inf, va[i])
        bi = jnp.where(b_done, neg_inf, vb[j])

        take_a = ai >= bi
        val = jnp.where(take_a, ai, bi)

        out = out.at[t].set(val)
        i = jnp.where(take_a & (~a_done), i + 1, i)
        j = jnp.where((~take_a) & (~b_done), j + 1, j)
        t = t + 1
        return (i, j, t, out)

    def cond(carry):
        _, _, t, _ = carry
        return t < K

    out = jnp.full((k,), neg_inf, dtype=a.dtype)
    carry0 = (0, 0, 0, out)
    _, _, _, out = lax.while_loop(cond, body, carry0)
    return out[:K]

# JIT-compiled versions for benchmarking — make k static (arg index 2)
jit_full   = jax.jit(topk_concat_full, static_argnums=(2,))
jit_2k     = jax.jit(topk_concat_2k,   static_argnums=(2,))
jit_stream = jax.jit(topk_streaming,   static_argnums=(2,))

# ----------------------------
# Benchmark harness
# ----------------------------

@dataclass
class Case:
    n_a: int
    n_b: int
    k: int
    dtype: jnp.dtype = jnp.float32

def _gen_inputs(n_a, n_b, dtype, key):
    k1, k2 = jax.random.split(key)
    a = jax.random.normal(k1, (n_a,), dtype=dtype)
    b = jax.random.normal(k2, (n_b,), dtype=dtype)
    return a, b

def _time_fn(fn, a, b, k, iters: int):
    _ = fn(a, b, k).block_until_ready()  # warmup
    t0 = time.perf_counter()
    for _ in range(iters):
        out = fn(a, b, k)
    out.block_until_ready()
    t1 = time.perf_counter()
    return (t1 - t0) / iters

def _check_equal(values_list, atol=0.0, rtol=0.0):
    ref = values_list[0]
    for v in values_list[1:]:
        if ref.shape != v.shape or not jnp.allclose(ref, v, atol=atol, rtol=rtol):
            return False
    return True

def main():
    CASES = [
        Case(100_000, 100_000, 10),
        Case(1_000_000, 1_000_000, 10),
        Case(1_000_000, 1_000_000, 100),
        Case(5_000_000, 5_000_000, 100),
        Case(2_000_000, 2_000_000, 5_000),
    ]

    ITERS = 30
    key = jax.random.PRNGKey(0)

    print(f"Device: {jax.devices()[0]}\n")

    for idx, case in enumerate(CASES, 1):
        k1, key = jax.random.split(key)
        a, b = _gen_inputs(case.n_a, case.n_b, case.dtype, k1)

        # Trigger compilation once per shape for each method
        _ = jit_full(a, b, case.k).block_until_ready()
        _ = jit_2k(a, b, case.k).block_until_ready()
        _ = jit_stream(a, b, case.k).block_until_ready()

        # Correctness
        v_full   = jit_full(a, b, case.k);   v_full.block_until_ready()
        v_2k     = jit_2k(a, b, case.k);     v_2k.block_until_ready()
        v_stream = jit_stream(a, b, case.k); v_stream.block_until_ready()
        ok = _check_equal([v_full, v_2k, v_stream])
        if not ok:
            print(f"[Case {idx}] WARNING: value mismatch (full vs 2k vs streaming).")

        # Timing
        t_full   = _time_fn(jit_full, a, b, case.k, ITERS)
        t_2k     = _time_fn(jit_2k, a, b, case.k, ITERS)
        t_stream = _time_fn(jit_stream, a, b, case.k, ITERS)

        na, nb, k = case.n_a, case.n_b, case.k
        print(f"[Case {idx}] n_a={na:,} n_b={nb:,} k={k:,} dtype={jnp.dtype(case.dtype).name}")
        print(f"  full-concat+top_k : {t_full*1e6:9.1f} µs  (baseline 1.00x)")
        print(f"  2k-concat+top_k   : {t_2k*1e6:9.1f} µs  (x{t_2k/t_full:0.2f})")
        print(f"  streaming-merge   : {t_stream*1e6:9.1f} µs  (x{t_stream/t_full:0.2f})\n")

if __name__ == "__main__":
    main()
