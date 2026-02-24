"""
Assignment 3

Task to pass with Grade 5:
1.	Research and understand various feature selection techniques, such as:
	a) Filter methods (e.g., Pearson correlation, chi-square test).
    b) Wrapper methods (e.g., recursive feature elimination).
    c) Embedded methods (e.g., LASSO, feature importance in tree-based models).
2.	Implement at least four feature selection algorithms in this project, applying them to the dataset.
"""
import numpy as np
import pandas as pd
import importlib
import Assignment_3_Grade_3 as etl  
import feature_filters as filters 
import feature_wrappers as wrappers
import feature_embedded as embedded

importlib.reload(embedded)                      # bugg 
print("embedded fil", embedded.__file__)        # bugg 


# För Grade 5  
from sklearn.svm import SVC
from feature_wrappers import Wrapper_Sfs, select_features_wrapper_sfs
from feature_embedded import Embedded_f1, select_features_embedded_l1
import Assignment_3_Grade_4 as g4



LABEL_COL = "event"   

#%% Load data 

### laddar det normaliserade datat från grade 3
#Xdf = etl.load_dataset("combined_dataset_normalized.csv")
#y = Xdf[LABEL_COL].values
    
df = etl.load_and_combine_csv()
y = df[LABEL_COL].values
y = etl.to_binary_event(df[LABEL_COL]).values # ger vaning om muilticlass mode, binäriserar datat 
Xdf = etl.drop_columns_safely(df)

# Normalsering av data sker i stödfunktionerna eftersom Filter Chi2 behöver MinMax skalning

#%%
# =========================================================
# 1a) Filters 
# =========================================================
filters.eval_filters(df,Xdf,y)

#%%
# debug check 
print("Class distribution (0=normal,1=event):", np.bincount(y))
print("Unique classes in y:", np.unique(y)[:10])
print("Number of classes:", len(np.unique(y)))
print("Number of samples:", len(y))

#%%
# =========================================================
# 1b) Wrappers
# =========================================================

print("Calculating wrappers, sit back, this takes a while ~6-8 min")
wrappers.eval_wrappers(Xdf, y)
    

#%%
# =========================================================
# 1c) Wrappers
# =========================================================

print("Embedded methods ")

#debug 

print("NaN kvar:", Xdf.isna().any().any())
print("Icke-numeriska kolumner:", list(Xdf.columns[Xdf.dtypes == "object"]))

embedded.eval_embedded_methods(Xdf, y)

# =========================================================
# 2) Implement algorithm och jämföra med tidigare resultat
# =========================================================

  # för Grade 5 
  # # 1) Läs data
df = g4.load_dataset("combined_dataset_normalized.csv")

# 2) Features + labels
X, y, feat_cols = g4.get_X_y(df, label_col="label")

# 3) PCA-2D
pca, X2 = g4.fit_pca_2d(X, seed=42)

# 6) Träna SVM med 5-fold tuning – använder make_kfold
svm_cv, best_params = g4.train_svm_kfold(X2, y, k=5, seed=42)

# 6b) Träna wrapper + embedded (på samma PCA-2D X2 för minimal förändring)
wrapper_sfs = Wrapper_Sfs(X2, y, best_params)
embedded_f1 = Embedded_f1(X2, y, best_params)

# --- Bygg DataFrame för originalfeatures ---
Xdf = pd.DataFrame(X, columns=feat_cols)

# ===== 1) Bas (CV-tunad SVM) på PCA från ALLA features =====
pca_all, X2_all = g4.fit_pca_2d(X, seed=42)
km_all, kmlab_all = g4.fit_kmeans(X2_all, n_clusters=2, seed=42)

svm_all = SVC(kernel="rbf", C=best_params["C"], gamma=best_params["gamma"], random_state=42, class_weight="balanced")
svm_all.fit(X2_all, y)

# ===== 2) Wrapper (SFS) selection -> PCA -> SVM =====
feats_wrap = select_features_wrapper_sfs(Xdf, y, best_params, k=15, scoring="f1", seed=42)
X_wrap = Xdf[feats_wrap].to_numpy(dtype=float)

pca_wrap, X2_wrap = g4.fit_pca_2d(X_wrap, seed=42)
km_wrap, kmlab_wrap = g4.fit_kmeans(X2_wrap, n_clusters=2, seed=42)

svm_wrap = SVC(kernel="rbf", C=best_params["C"], gamma=best_params["gamma"], random_state=42, class_weight="balanced")
svm_wrap.fit(X2_wrap, y)

# ===== 3) Embedded (L1) selection -> PCA -> SVM =====
feats_emb = select_features_embedded_l1(Xdf, y, max_features=15, seed=42)
X_emb = Xdf[feats_emb].to_numpy(dtype=float)

pca_emb, X2_emb = g4.fit_pca_2d(X_emb, seed=42)
km_emb, kmlab_emb = g4.fit_kmeans(X2_emb, n_clusters=2, seed=42)

svm_emb = SVC(kernel="rbf", C=best_params["C"], gamma=best_params["gamma"], random_state=42, class_weight="balanced")
svm_emb.fit(X2_emb, y)

# ===== Plot 3 paneler likt Grade 4 (3 olika svärmar) =====
g4.plot_side_by_side_compare_A(
    X2_list=[X2_all, X2_wrap, X2_emb],
    y=y,
    kmlab_list=[kmlab_all, kmlab_wrap, kmlab_emb],
    models=[svm_all, svm_wrap, svm_emb],
    titles=[
        f"SVM (CV-tunad, 5-fold) + marginaler\nbäst: {best_params}",
        f"Wrapper Forward SFS -> PCA-2D (k={len(feats_wrap)})",
        f"Embedded L1 -> PCA-2D (k={len(feats_emb)})"
    ],
    save_path="svm_side_by_side_compare_A.png",
    show=True,
    draw_svm_margins=True
)

print("Wrapper valda features:", feats_wrap)
print("Embedded valda features:", feats_emb)




#%% 
