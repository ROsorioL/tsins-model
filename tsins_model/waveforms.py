"""
Stimulation waveform synthesis: TSINS (Osorio et al., Eqs. 1-6), TIS and the
rectangular biphasic control.

All channel waveforms are dimensionless and scaled by the nominal amplitude in
mA, matching the experiment, where the DS5 delivers 1 mA per 1 V of command and
the synthesised waveform is multiplied by the nominal setting.
"""
import numpy as np

F0_DEFAULT = 2.0          # Hz, fundamental of the target waveform
N0_DEFAULT = 4            # first harmonic
K_DEFAULT = 7             # harmonics per channel
KAPPA_DEFAULT = 10.0      # blocking amplitude / max harmonic amplitude
G_DEFAULT = 1.0 / 16.0    # global gain


def tsins_channels(t, fb, f0=F0_DEFAULT, n0=N0_DEFAULT, K=K_DEFAULT,
                   kappa=KAPPA_DEFAULT, g=G_DEFAULT, An=None, phin=None):
    """Eqs. (1)-(2): per-channel harmonic subset plus the antiphase blocker.

    Returns dict with the low-frequency (harmonic) and kilohertz (blocking)
    parts of each channel kept separate, so that a frequency-dependent volume
    conductor can be applied band by band.
    """
    t = np.asarray(t, float)
    n = np.arange(n0, n0 + 2 * K)
    An = np.ones(n.size) if An is None else np.asarray(An, float)
    phin = np.zeros(n.size) if phin is None else np.asarray(phin, float)
    Ab = kappa * An.max()
    h = An[:, None] * np.sin(2 * np.pi * n[:, None] * f0 * t[None, :] + phin[:, None])
    ch1_lf = g * h[:K].sum(axis=0)
    ch2_lf = g * h[K:].sum(axis=0)
    ch1_hf = g * Ab * np.sin(2 * np.pi * fb * t)
    ch2_hf = g * Ab * np.sin(2 * np.pi * fb * t + np.pi)
    return dict(ch1_lf=ch1_lf, ch1_hf=ch1_hf, ch2_lf=ch2_lf, ch2_hf=ch2_hf,
                target=g * h.sum(axis=0), harmonics=n, Ab=Ab, g=g)


def tis_channels(t, fb, df=2.0, amp=1.0):
    """Two kHz carriers offset by df; the beat is the amplitude modulation."""
    t = np.asarray(t, float)
    z = np.zeros_like(t)
    return dict(ch1_lf=z, ch1_hf=amp * np.sin(2 * np.pi * fb * t),
                ch2_lf=z, ch2_hf=amp * np.sin(2 * np.pi * (fb + df) * t),
                target=amp * (np.sin(2 * np.pi * fb * t)
                              + np.sin(2 * np.pi * (fb + df) * t)))


def biphasic_channels(t, rate=2.0, pw_ms=0.25, t0=None, band_cut=500.0):
    """Charge-balanced rectangular biphasic pulses, identical on both channels.

    Figure 8 of the paper shows both channels carrying the control pulse in
    phase, so the in-tissue sum is twice one channel.
    """
    t = np.asarray(t, float)
    pw = pw_ms * 1e-3
    period = 1.0 / rate
    t0 = 0.5 * period if t0 is None else t0
    ph = np.mod(t - t0, period)
    w = np.where(ph < pw, 1.0, np.where(ph < 2 * pw, -1.0, 0.0))
    lf, hf = split_bands(t, w, band_cut)
    return dict(ch1_lf=lf, ch1_hf=hf, ch2_lf=lf.copy(), ch2_hf=hf.copy(),
                target=2 * w)


def split_bands(t, sig, f_cut=500.0):
    """Split a signal into components below and above f_cut (rFFT, brick wall)."""
    n = sig.size
    dt = t[1] - t[0]
    S = np.fft.rfft(sig)
    f = np.fft.rfftfreq(n, dt)
    lo = S.copy(); lo[f >= f_cut] = 0.0
    hi = S.copy(); hi[f < f_cut] = 0.0
    return np.fft.irfft(lo, n), np.fft.irfft(hi, n)


# --------------------------------------------------------------------------
def make_drive(waveform, t, fb, amp_mA, f0=F0_DEFAULT, df=2.0, rate=2.0,
               pw_ms=0.25, ramp_ms=0.0, **kw):
    """Return (ivec, labels) with ivec[:, k] in mA for the four lead-field terms.

    labels order: ch1_lf, ch1_hf, ch2_lf, ch2_hf
    """
    if waveform == "tsins":
        d = tsins_channels(t, fb, f0=f0, **kw)
    elif waveform == "tis":
        d = tis_channels(t, fb, df=df)
    elif waveform == "biphasic":
        d = biphasic_channels(t, rate=rate, pw_ms=pw_ms)
    else:
        raise ValueError(waveform)
    keys = ["ch1_lf", "ch1_hf", "ch2_lf", "ch2_hf"]
    ivec = np.column_stack([d[k] for k in keys]) * amp_mA
    if ramp_ms > 0:
        env = np.clip(t / (ramp_ms * 1e-3), 0, 1)[:, None]
        ivec = ivec * env
    return ivec, keys, d


def channel_sum(ivec):
    """Total current per channel (mA) and the linear in-tissue sum."""
    ch1 = ivec[:, 0] + ivec[:, 1]
    ch2 = ivec[:, 2] + ivec[:, 3]
    return ch1, ch2, ch1 + ch2


def delivered_charge(t, ivec):
    """Q = int|I1| + int|I2| over the window, in mC (t in s, I in mA)."""
    ch1, ch2, _ = channel_sum(ivec)
    dt = t[1] - t[0]
    return (np.abs(ch1).sum() + np.abs(ch2).sum()) * dt
