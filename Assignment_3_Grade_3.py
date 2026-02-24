"""
Assignment 3

Task to pass with Grade 3:
1.	Data preprocessing:
	Load the data from all three files.
	Combine the three datasets into a single unified dataset.
	Remove the columns start_time, axle, cluster, tsne_1, and tsne_2 from the dataset.
	Replace all normal events with 0 and all other events with 1.
2.	Data transformation:
    Normalize the dataset.
        Edit: By Standaization or min-max. First one chosen since it works well with Support Vector Machine (SVM) 

"""

# Imports 

import numpy as np
import pandas as pd
from pathlib import Path


# 1 Data processing ###########################################################################################

# === Anpassa vid behov ===
# Kolumnnamnet som innehåller event/etiketter (t.ex. 'event', 'label', 'klass')
EVENT_COL = "event"  

# Filnamnen 
files = [
    "Trail1_extracted_features_acceleration_m1ai1.csv",
    "Trail2_extracted_features_acceleration_m1ai1.csv",
    "Trail3_extracted_features_acceleration_m2ai0.csv",
]

# Kolumner som ska tas bort om de finns, enligt instruktion 
COLS_TO_DROP = ["start_time", "axle", "cluster", "tsne_1", "tsne_2"]

#För att preppa datat så man kan använda "Assignemt 3 Grade 3 Plot.py"
#COLS_TO_DROP = ["axle", "cluster", "tsne_1", "tsne_2"]  # ← behåller start_tim, för att kunna plotta diagram som i PPTn 

# Filväg för export av sammanfogad data
OUTPUT_CSV = "combined_dataset.csv"
OUTPUT_NORMALIZED_CSV = "combined_dataset_normalized.csv"

# =========================================================
# Stöd funktioner
# =========================================================

#%% 
# stödfunktion som kan användas senare
def load_dataset(csv_path=OUTPUT_NORMALIZED_CSV, nrows=None):
    return pd.read_csv(csv_path, nrows=nrows)

#%%
def load_and_combine_csv(file_list=files):
    """Läs in flera CSV-filer och kombinera dem vertikalt (radvis)."""
    dfs = []
    for f in file_list:
        p = Path(f)
        if not p.exists():
            print(f"[Varning] Filen hittades inte: {p.resolve()}")
            continue
        try:
            df = pd.read_csv(p)
            dfs.append(df)
            print(f"[OK] Läste in {p.name} med form {df.shape}")
        except Exception as e:
            print(f"[Fel] Kunde inte läsa {p.name}: {e}")
    if not dfs:
        raise FileNotFoundError("Inga giltiga CSV-filer kunde läsas in.")
    return pd.concat(dfs, ignore_index=True)

#%%
def drop_columns_safely(df, cols=COLS_TO_DROP):
    """Ta bort angivna kolumner om de finns."""
    existing = [c for c in cols if c in df.columns]
    if existing:
        df = df.drop(columns=existing, errors="ignore")
        print(f"[Info] Tog bort kolumner: {existing}")
    else:
        print("[Info] Inga av de angivna kolumnerna fanns att ta bort.")
    return df


#%%
def to_binary_event(series):
    """
    Mappa 'normala' event -> 0, övriga -> 1.

    - Om värdet är tal: 0 hålls som 0, annat blir 1.
    - Om text: om den innehåller 'normal' -> 0, annars 1.
    """
    def map_one(x):    # x ett värde från kolumnen som skickats in via series 
        # Numeriska fall
        try:
            # Försök tolka som tal
            xv = float(x)
            return 0 if xv == 0 else 1
        except Exception:
            pass

        # Textuella fall
        if pd.isna(x):
            return 1  # behandla NaN som "icke-normal" för säkerhets skull
        
        # konverterar till sträng, tar bort whitespace och konvertyerar till lowercase för att kunna testa
        s = str(x).strip().lower()

        # testar om eventet är 'normal' 
        if s == "normal":
            return 0
  
        return 1

    return series.apply(map_one) # kolumn från dataframe. funktinoen map_one kör varje värde på kolumnen. 



#%%
# =========================================================
# 2. Normalisering av data
# =========================================================

def normalize_and_save(
    input_csv=OUTPUT_CSV ,
    output_csv=OUTPUT_NORMALIZED_CSV,
    exclude_cols=None,
    method="zscore", # också kallas standadiserad 
):
    # Läs
    df = pd.read_csv(input_csv)

    # Om event_str saknas (t.ex. från äldre körning): skapa från 'event' om den finns
    if "event_str" not in df.columns and "event" in df.columns:
        df["event_str"] = df["event"].astype(str)

    # Se till att vi INTE skalar följande kolumner
    exclude_cols = exclude_cols 

    # Vilka numeriska kolumner ska skalas?

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cols_to_scale = [c for c in numeric_cols if c not in exclude_cols]

    # egentligen onödigt men koll på att vi verkligen har kolumner som vi ska skala. 
    if not cols_to_scale:
        print("[Info] Inga numeriska kolumner att normalisera (utöver exkluderade).")
        df.to_csv(output_csv, index=False, encoding="utf-8")
        print(f"[Klart] Sparade (oförändrat) till: {output_csv}")
        return

    df_scaled = df.copy()

    # standadiserad , bättre vid SVM 
    if method == "zscore":
        for c in cols_to_scale:
            col = df_scaled[c].astype(float)
            mean = col.mean()
            std = col.std(ddof=0) # degrees of freedom=0,  √( Σ (x–μ)² / n )  Population standard deviation 
            if std == 0 or np.isnan(std):
                print(f"[Info] Hoppar över z-score på konstant/NaN-kolumn: {c}")
                continue
            df_scaled[c] = (col - mean) / std   

    # min max, Är mindre lämplig vid SVM 
    elif method == "minmax":
        for c in cols_to_scale:
            col = df_scaled[c].astype(float)
            cmin, cmax = col.min(), col.max()
            if cmax == cmin or np.isnan(cmin) or np.isnan(cmax):
                print(f"[Info] Hoppar över min-max på konstant/NaN-kolumn: {c}")
                continue
            df_scaled[c] = (col - cmin) / (cmax - cmin)
    else:
        raise ValueError("Okänd metod. Använd 'zscore' eller 'minmax'.")

    df_scaled.to_csv(output_csv, index=False, encoding="utf-8")
    print(f"[Klart] Sparade normaliserad data till: {output_csv}")




#%%
# main ########################################################################################

def main():
    # 1) Läs och kombinera
    df = load_and_combine_csv(files)

    # 2) Ta bort oönskade kolumner
    df = drop_columns_safely(df, COLS_TO_DROP)

    # 3) Spara undan eventtexten i en egen kolumn för plotting
    if EVENT_COL not in df.columns:
        raise KeyError(
            f"Event-kolumnen '{EVENT_COL}' finns inte i data. "
            f"Tillgängliga kolumner är: {list(df.columns)}.\n"
            f"Ändra variabeln EVENT_COL högst upp i scriptet till rätt kolumnnamn."
        )

    # behåll originaltext för etiketter
    df["event_str"] = df[EVENT_COL].astype(str)

    # 4) Skapa label som NY kolumn (0=normal, 1=icke-normal)
    df["label"] = to_binary_event(df[EVENT_COL])

    # (viktigt) Behåll 'event' kvar – vi använder event_str/ev. event senare i plottar

    # 5) Spara resultat
    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")
    print(f"[Klart] Sparade kombinerad och rensad data till: {OUTPUT_CSV}")
    print("[Fördelning] label-värden:\n", df["label"].value_counts(dropna=False))


    # 6) Normalizera resultatet 
    normalize_and_save(
        input_csv=OUTPUT_CSV,                      # combined_dataset.csv
        output_csv=OUTPUT_NORMALIZED_CSV,
        exclude_cols=["label", "start_time", "event"],      # ska INTE normalizera dessa kolumner 
        method="zscore"                             # eller "minmax"
    )
 
    #debug 

    print(load_dataset(OUTPUT_NORMALIZED_CSV, 0).columns.tolist())
    # Borde se 'start_time', 'label', 'event', 'event_str', plus alla features



#%%
# Om man enbart vill köra stand alone, inssåg att jag kunde återanvända i grade 5
if __name__ == "__main__":
    main()