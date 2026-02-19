import numpy as np

def otsu_threshold_1d(x, nbins=512):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    hist, edges = np.histogram(x, bins=nbins)
    p = hist.astype(float) / (hist.sum() + 1e-12)
    centers = (edges[:-1] + edges[1:]) / 2

    w0 = np.cumsum(p)
    w1 = 1 - w0
    mu = np.cumsum(p * centers)
    mu_t = mu[-1]

    denom = w0 * w1
    denom[denom == 0] = np.nan
    sigma_b2 = (mu_t * w0 - mu) ** 2 / denom

    idx = np.nanargmax(sigma_b2)
    return float(centers[idx])

def three_state_from_otsu_deadband(S, delta_mode="idle_std", k=1.5, nbins=512):
    """
    S: (T,) sum-of-capacitance (>=0 권장)
    Returns:
      state: int array (0=idle, 1=uncertain, 2=touch)
      thresholds in original domain: (T_idle, T_touch)
      thresholds in log domain: (t0, delta)
    """
    S = np.asarray(S, dtype=float)
    S = np.maximum(S, 0)
    L = np.log1p(S)

    t0 = otsu_threshold_1d(L, nbins=nbins)

    # delta 결정
    if delta_mode == "idle_std":
        idle_L = L[L < t0]
        if len(idle_L) < 10:
            # fallback: 전체 std의 일부
            delta = k * (np.std(L) + 1e-8) * 0.2
        else:
            delta = k * (np.std(idle_L) + 1e-8)
    else:
        raise ValueError("delta_mode currently supports only 'idle_std'")

    t_idle = t0 - delta
    t_touch = t0 + delta

    state = np.full_like(L, 1, dtype=np.int32)  # 기본 uncertain
    state[L < t_idle] = 0
    state[L > t_touch] = 2

    T_idle = float(np.expm1(t_idle))
    T_touch = float(np.expm1(t_touch))

    return state, (T_idle, T_touch), (t0, delta)
