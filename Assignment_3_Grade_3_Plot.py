#Assignment 3


# Imports 

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

#%%
# Plotta översikt av normaliserat data ############################################# 


def plot_normalized_overview(
    input_csv="combined_dataset_normalized.csv",
    label_col="label",
    time_col="start_time",
    save_prefix="normalized_view_simple",
    show=True
):
    """
    Enkel översikt:
      A) PCA (2D) färgad efter label (0=grå, 1=röd)
      B) Mahalanobis-distans i PCA-2D, plottad över tid (med P95-tröskel)

    Antaganden:
      - input_csv innehåller 'label' (0/1)
      - 'start_time' finns (sekunder, oförändrad vid normalisering)
      - övriga numeriska kolumner är normaliserade features
    """

    # --- 0) Läs & sanity ---
    df = pd.read_csv(input_csv)
    if label_col not in df.columns:
        raise KeyError(f"'{label_col}' saknas i {input_csv}. Kolumner: {list(df.columns)}")

    # Tidsaxel (sekunder → DatetimeIndex)
    if time_col in df.columns:
        t_s = df[time_col].astype(float).to_numpy()
    else:
        print(f"[Info] '{time_col}' saknas – använder sekventiell tid.")
        t_s = np.arange(len(df), dtype=float)

    # robust, enkel konvertering till datetime (undviker DatetimeArray + Timestamp-fel)
    t_idx = pd.to_datetime(np.asarray(t_s, dtype=float), unit="s", origin="unix")

    # --- 1) Featurematris (alla numeriska kolumner, exkludera label + tid) ---
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    feat_cols = [c for c in num_cols if c not in (label_col, time_col)]
    if not feat_cols:
        raise ValueError("Hittar inga numeriska feature-kolumner att använda (utöver label/tid).")

    X = df[feat_cols].to_numpy(dtype=float)
    y = df[label_col].astype(int).to_numpy()

    # --- 2) PCA (2D) via SVD (enkelt och snabbt) ---
    Xc = X - X.mean(axis=0, keepdims=True)    # centrera
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    Z2 = U[:, :2] * S[:2]                      # (n, 2)
    z1, z2 = Z2[:, 0], Z2[:, 1]

    # --- 3) Mahalanobis-distans i PCA-2D (litet & stabilt) ---
    cov = np.cov(Z2.T)                         # (2,2)
    cov_inv = np.linalg.pinv(cov)              # pseudo-invers för robusthet
    mu = Z2.mean(axis=0, keepdims=True)
    diffs = Z2 - mu
    md2 = np.einsum("ij,jk,ik->i", diffs, cov_inv, diffs)
    md = np.sqrt(np.maximum(md2, 0))

    # Skala md till [0,1] så variationen syns tydligt (enkelt och visuellt)
    md_min, md_max = md.min(), md.max()
    md_plot = (md - md_min) / (md_max - md_min + 1e-12)
    thr = np.percentile(md_plot, 95)  # P95 som enkel tröskel

    is_abn = (y == 1)
    is_norm = ~is_abn

    # --- 4A) PCA-plot (2D), färg efter label ---
    fig1, ax1 = plt.subplots(figsize=(7.5, 6))
    ax1.scatter(z1[is_norm], z2[is_norm], s=10, c="gray", alpha=0.25, label="Normal (0)")
    ax1.scatter(z1[is_abn], z2[is_abn], s=24, c="crimson", alpha=0.9, label="Icke-normal (1)")
    ax1.set_title("PCA (2D) av normaliserade features")
    ax1.set_xlabel("PC1"); ax1.set_ylabel("PC2")
    ax1.grid(True, alpha=0.2); ax1.legend(loc="best", frameon=False)
    fig1.tight_layout()
    fig1.savefig(f"{save_prefix}_pca2d.png", dpi=150, bbox_inches="tight")
    print(f"[Sparat] {save_prefix}_pca2d.png")

    # --- 4B) Mahalanobis över tid (i PCA-2D) ---
    # sortera på tid för snygg linje
    order = np.argsort(t_idx.values)
    t_plot = t_idx.values[order]
    mdp_plot = md_plot[order]
    is_abn_sorted = is_abn[order]

    fig2, ax2 = plt.subplots(figsize=(10, 3.2))
    ax2.plot(t_plot, mdp_plot, color="tab:blue", lw=1.2, label="Mahalanobis (PCA-2D, 0–1)")
    ax2.scatter(t_plot[is_abn_sorted], mdp_plot[is_abn_sorted], s=20, c="crimson", label="Icke-normal (1)")
    ax2.axhline(thr, color="orange", ls="--", lw=1.1, label=f"P95 ≈ {thr:.2f}")
    ax2.set_title("Anomali‑score över tid (Mahalanobis i PCA‑2D)")
    ax2.set_xlabel("Tid"); ax2.set_ylabel("Distans (0–1)")
    ax2.grid(True, alpha=0.2); ax2.legend(loc="best", frameon=False)
    fig2.tight_layout()
    fig2.savefig(f"{save_prefix}_mahalanobis_time.png", dpi=150, bbox_inches="tight")
    print(f"[Sparat] {save_prefix}_mahalanobis_time.png")

    if show:
        plt.show()



#%% 
# plotta som exemplet i kursmaterialet  ################################ ##############


def plot_acceleration_with_events(
    df,
    time_col="start_time",
    accel_col="max",        # vilken kolumn som representerar "acceleration"
    event_col="event_str",  # original event-string
    label_col="label",      # 0/1
    title="Acceleration över tid med event-markeringar",
    save_path=None,
    show=True
):
    """
    Plotta en 'pseudo-accelerationssignal' över tid genom att använda t.ex.
    kolumnen 'max' eller 'rms' som proxy för toppar i råsignal.
    Markera events med vertikala linjer + etiketter.
    """

    # Säkerställ sortering i tidsordning
    df = df.sort_values(time_col).reset_index(drop=True)

    times = df[time_col].to_numpy()
    accel = df[accel_col].to_numpy()
    events = df[event_col].astype(str).to_numpy()
    labels = df[label_col].astype(int).to_numpy()

    fig, ax = plt.subplots(figsize=(14, 5))

    # --- plot "acceleration" (proxy från features)
    ax.plot(times, accel, color="tab:blue", lw=1)
    ax.set_xlabel("tid i sekunder")
    ax.set_ylabel(f"{accel_col} (proxy för acceleration)")
    ax.set_title(title)
    ax.grid(True, alpha=0.2)

    # --- markera event med vertikala linjer
    for t, ev, lab in zip(times, events, labels):
        color = "crimson" if lab == 1 else "gray"
        alpha = 0.9 if lab == 1 else 0.3

        ax.axvline(x=t, color=color, alpha=alpha, linewidth=1)

        # etikett under x-axeln
        ax.text(
            t,               # x-position
            ax.get_ylim()[0] - (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.08,
            ev,
            rotation=90,
            ha="center",
            va="top",
            fontsize=8,
            alpha=alpha,
            color=color
        )

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[Sparat] {save_path}")

    if show:
        plt.show()

    return fig, ax

#%%
# plotta normaliserat data ################################## #######################



def plot_normalized_time_series_with_events(
    input_csv="combined_dataset_normalized.csv",
    time_col="start_time",
    series_cols=("max", "min", "rms"),
    event_col="event_str",
    label_col="label",
    title_prefix="Acceleration (proxy) över tid – normaliserat data",
    save_path="normalized_accel_with_events.png",
    show=True,
    annotate_only_abnormal=True,   # <-- NY: skriv ut text bara för onormala events
):

    """
    Läser ett *normaliserat* dataset med start_time och plottar tre valda tidsserier
    (t.ex. max, min, rms) mot tid. Vertikala linjer markerar event; etiketter under x-axeln.
    """

    p = Path(input_csv)
    if not p.exists():
        raise FileNotFoundError(f"Hittar inte filen: {p.resolve()}")

    df = pd.read_csv(p)

    # Kolla att nödvändiga kolumner finns
    missing = [c for c in [time_col, label_col] + list(series_cols) if c not in df.columns]
    if missing:
        raise KeyError(
            f"Kolumner saknas i {input_csv}: {missing}\n"
            f"Tillgängliga kolumner: {list(df.columns)}"
        )

    # Om event_str inte finns men 'event' finns: skapa den
    if event_col not in df.columns and "event" in df.columns:
        df["event_str"] = df["event"].astype(str)
        event_col = "event_str"

    # Sortera på tid och förbered index
    df = df.sort_values(time_col).reset_index(drop=True)
    t = df[time_col].to_numpy()
    labels = df[label_col].astype(int).to_numpy()
    events = df[event_col].astype(str).to_numpy() if event_col in df.columns else np.array([""] * len(df))

    # Skapa figuren
    n_series = len(series_cols)
    fig, axes = plt.subplots(n_series, 1, figsize=(14, 3.5 * n_series), sharex=True)
    if n_series == 1:
        axes = [axes]


    for ax, col in zip(axes, series_cols):
        y = df[col].to_numpy(dtype=float)
        ax.plot(t, y, color="tab:blue", lw=1.2)
        ax.set_ylabel(f"{col} (norm.)")
        ax.grid(True, alpha=0.2)
        ax.set_title(f"{title_prefix}: {col}")

        # Vertikala linjer + etiketter
        y_min, y_max = np.min(y), np.max(y)
        label_y = y_min - (y_max - y_min) * 0.08 if y_max > y_min else y_min - 0.1

        for ti, ev, lab in zip(t, events, labels):
            # Linjer: röda för onormal (1), grå och svag för normal (0)
            color = "crimson" if lab == 1 else "gray"
            alpha = 0.95 if lab == 1 else 0.20
            lw    = 1.2 if lab == 1 else 0.8
            ax.axvline(ti, color=color, alpha=alpha, lw=lw)

            # Etikettera endast i nedersta axeln – och bara onormala om så önskas
            if ax is axes[-1] and ev:
                if (not annotate_only_abnormal) or (lab == 1):
                    axes[-1].text(
                        ti, label_y, ev, rotation=90, ha="center", va="top",
                        fontsize=8, color=color, alpha=0.9 if lab == 1 else 0.35
                    )


    axes[-1].set_xlabel("tid [s]")
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[Sparat] {save_path}")

    if show:
        plt.show()

    return fig, axes



############################################### main #########################################################

#%%
def main():
    
    # 7 ) plotta det normaliserade resutlatet 

    plot_normalized_overview(
        input_csv="combined_dataset_normalized.csv",
        label_col="label",
        time_col="start_time",
        save_prefix="normalized_view_simple",
        show=True
    )

    # Efter normalize_and_save(...)
    plot_normalized_time_series_with_events(
        input_csv="combined_dataset_normalized.csv",
        time_col="start_time",
        series_cols=("max", "min", "rms"),    # ändra gärna ordningen/valet
        event_col="event_str",                
        label_col="label",
        save_path="normalized_accel_with_events.png",
        show=True
    )


    # 8 ) plotta som exemplet i materialet 
    
    df = pd.read_csv("combined_dataset.csv")

    plot_acceleration_with_events(
        df,
        time_col="start_time",
        accel_col="max",         # eller "rms" om du vill jämna ut signalen
        event_col="event_str",   # original eventnamn
        label_col="label",
        save_path="accel_with_events.png",
    )



#%%
if __name__ == "__main__":
    main()