"""
Assignment 3

Task to pass with Grade 4:
1.	Data  set splitting:
	Split the data into training and testing sets in an 80/20 ratio.
2.	Cross-validation
    Perform k-fold cross-validation (e.g., 5-fold) on the training set to evaluate model stability.
3.  Comparison task: 
    Compare between the 80/20 train-test split and k-fold cross-validation using SVM (Support Vector Machine).  
    Train an SVM model using both methods and evaluate its performance. 
    Discuss the differences in accuracy, consistency of results, and generalization ability.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV,cross_val_predict, LeaveOneOut
from sklearn.metrics import accuracy_score
#from sklearn.neighbors import KNeighborsClassifier



#%%
# =========================================================
# 1) Läs data
# =========================================================
def load_dataset(csv_path="combined_dataset_normalized.csv"):
    return pd.read_csv(csv_path)

#%%
def get_X_y(df, label_col="label", drop_cols=("event", "event_str", "start_time")):
    """
    Returnerar (X, y, feature_cols).
    Tar numeriska kolumner och droppar label + ev drop_cols.
    """
    if label_col not in df.columns:
        raise KeyError(f"'{label_col}' saknas. Tillgängliga kolumner: {list(df.columns)}")

    # numeriska kolumner
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    # features = numeriska minus label minus drop_cols
    feat_cols = [c for c in num_cols if c != label_col and c not in drop_cols]
    if not feat_cols:
        raise ValueError("Inga numeriska feature-kolumner hittades (utöver label/drop_cols).")

    X = df[feat_cols].to_numpy(dtype=float)
    y = df[label_col].astype(int).to_numpy()
    return X, y, feat_cols

#%%
# =========================================================
# 2) Split-funktioner 
# =========================================================
def split_80_20(X, y, test_size=0.2, seed=42):
    #https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
    return train_test_split(
        X, y, test_size=test_size, stratify=y, shuffle=True, random_state=seed
    )

#%%
def make_kfold(y, k=5, seed=42):
    #https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedKFold.html#sklearn.model_selection.StratifiedKFold
    return StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)

#%%
# =========================================================
# 3) PCA + KMeans för visualisering
# =========================================================
def fit_pca_2d(X, seed=42):
    pca = PCA(n_components=2, random_state=seed)
    X2 = pca.fit_transform(X)
    return pca, X2

#%%
def fit_kmeans(X2, n_clusters=2, seed=42):
    km = KMeans(n_clusters=n_clusters, n_init="auto", random_state=seed)
    kmlab = km.fit_predict(X2)
    return km, kmlab

#%%
# =========================================================
# 4A) Träna SVM (80_20-out och k-fold)
# =========================================================
def train_svm_80_20(X2, y, test_size=0.2, seed=42, C=1.0, gamma="scale", class_weight=None):

    X_tr, X_te, y_tr, y_te = split_80_20(X2, y, test_size=test_size, seed=seed)

    #https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html
    svm = SVC(kernel="rbf", C=C, gamma=gamma, class_weight=class_weight, random_state=seed)
    svm.fit(X_tr, y_tr)

    return svm

#%%
def train_svm_kfold(X2, y, k=5, seed=42, class_weight=None):

    # https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html
    #
    # C styr balas mellan stor marginal och rätt klassificering 
    # gamma styr hur långt bort en punkt påverkar beslutsgränsen.
    #
    param_grid = {
        "C": [0.5, 1, 2, 5, 10],
        "gamma": ["scale", 0.1, 0.05, 0.01],
    } 

    base = SVC(kernel="rbf", class_weight=class_weight, random_state=seed)
    cv = make_kfold(y, k=k, seed=seed)

    gs = GridSearchCV(base, param_grid=param_grid, cv=cv, scoring="f1", n_jobs=None)
    gs.fit(X2, y)

    best_model = gs.best_estimator_     # tränad på hela X2 med bästa param
    best_params = gs.best_params_
    return best_model, best_params


#%%
# =========================================================
# 4B) Räkna ut error på varje modell 
# =========================================================
 
def compute_errors_80_20(svm_model, X2, y, test_size=0.2, seed=42):
    """
    Returnerar training_error och test_error för 80_20-out.
    training_error = 1 - accuracy på train
    test_error     = 1 - accuracy på test
    """
    X_tr, X_te, y_tr, y_te = train_test_split(
        X2, y, test_size=test_size, stratify=y, shuffle=True, random_state=seed
    )

    svm_model.fit(X_tr, y_tr)

    y_pred_tr = svm_model.predict(X_tr)
    y_pred_te = svm_model.predict(X_te)

    train_err = 1.0 - accuracy_score(y_tr, y_pred_tr)
    test_err  = 1.0 - accuracy_score(y_te, y_pred_te)

    return train_err, test_err
   

def compute_errors_kfold(svm_model, X2, y, k=5, seed=42):
    """
    Returnerar training_error och test_error för k-fold.
    test_error = OOF error (cross_val_predict)
    training_error = 1 - accuracy när modellen tränas på hela X2 och predikterar hela X2
    """
    cv = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)

    # Out-of-fold predictions (test per fold)
    y_pred_oof = cross_val_predict(svm_model, X2, y, cv=cv, method="predict")
    test_err = 1.0 - accuracy_score(y, y_pred_oof)

    # Train error (fit på hela)
    svm_model.fit(X2, y)
    y_pred_all = svm_model.predict(X2)
    train_err = 1.0 - accuracy_score(y, y_pred_all)

    return train_err, test_err

    


#%%
# =========================================================
# 5) Meshgrid + boundaries (KMeans & SVM margins)
# =========================================================
def make_meshgrid(X2, pad=0.6, step=0.02):
    x_min, x_max = X2[:, 0].min() - pad, X2[:, 0].max() + pad
    y_min, y_max = X2[:, 1].min() - pad, X2[:, 1].max() + pad
    xx, yy = np.meshgrid(np.arange(x_min, x_max, step),
                         np.arange(y_min, y_max, step))
    grid = np.c_[xx.ravel(), yy.ravel()]
    return xx, yy, grid



#%%
def svm_margin_lines(ax, svm_model, xx, yy, grid,
                     draw_boundary=True, draw_margins=True,
                     boundary_color="black", boundary_width=1.0, boundary_alpha=0.9,
                     margin_color="black", margin_style="--", margin_width=0.5, margin_alpha=0.7,
                     draw_support_vectors=True, sv_edge_color="black", sv_size=70, sv_linewidth=1.2, sv_alpha=0.9):
    """
    Ritar SVM decision boundary + margins (+ support vectors).
    Funkar för RBF också (gräns blir kurvad).
    https://www.quarkml.com/2022/10/the-rbf-kernel-in-svm-complete-guide.html
    """
    decision = svm_model.decision_function(grid).reshape(xx.shape)

    if draw_boundary:
        ax.contour(xx, yy, decision, levels=[0], colors=boundary_color,
                   linewidths=boundary_width, alpha=boundary_alpha)

    if draw_margins:
        ax.contour(xx, yy, decision, levels=[-1, 1], colors=margin_color,
                   linestyles=margin_style, linewidths=margin_width, alpha=margin_alpha)

    if draw_support_vectors and hasattr(svm_model, "support_vectors_"):
        sv = svm_model.support_vectors_
        ax.scatter(sv[:, 0], sv[:, 1], s=sv_size, facecolors="none",
                   edgecolors=sv_edge_color, linewidths=sv_linewidth, alpha=sv_alpha,
                   label="Support vectors")

#%%
def add_error_box(ax, training_error, test_error, gap, loc="lower left"):
    """
    Lägger en textbox i en matplotlib-ax med training/test/bayes error.
    """
    lines = [
        f"Training Error: {training_error:.3f}",
        f"Test Error:     {test_error:.3f}",
        f"Gap:     {gap:.3f}",
    ]


    text = "\n".join(lines)

    # Placera i hörn med samma "look" som exemplet
    ax.text(
        0.03, 0.03, text,
        transform=ax.transAxes,
        fontsize=9,
        va="bottom", ha="left",
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white", alpha=0.85, edgecolor="gray")
    )
#%%
# =========================================================
# 6) Side-by-side plot 
# =========================================================
def plot_side_by_side(
    X2, y, kmlab, km, svm_80_20, svm_cv, best_params,
    save_path="svm_side_by_side.png",
    show=True,
    draw_svm_margins=True
):
    xx, yy, grid = make_meshgrid(X2, pad=0.6, step=0.02)

    # beslutsytor (b) – samma som du hade
    Z_80_20 = svm_80_20.predict(grid).reshape(xx.shape)
    Z_cv = svm_cv.predict(grid).reshape(xx.shape)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharex=True, sharey=True)


    for ax, model, Z, title in [
        (axes[0], svm_80_20, Z_80_20, "SVM (80/20) + marginaler(PCA-2D)"),
        (axes[1], svm_cv,   Z_cv,   f"SVM (CV-tunad, 5-fold) + marginaler \nbäst: {best_params}")
    ]:
        # SVM bakgrund
        ax.contourf(xx, yy, Z, alpha=0.18, cmap="coolwarm")

        # lägger till fel ruta 
        
        if ax is axes[0]:
            tr_err, te_err = compute_errors_80_20(
                SVC(kernel="rbf", C=1.0, gamma="scale", random_state=42),
                X2, y, test_size=0.2, seed=42
            )
            add_error_box(ax, tr_err, te_err,(te_err-tr_err))

        if ax is axes[1]:
            tr_err, te_err = compute_errors_kfold(
                SVC(kernel="rbf", C=best_params["C"], gamma=best_params["gamma"], random_state=42),
                X2, y, k=5, seed=42
            )
            add_error_box(ax, tr_err, te_err, (te_err-tr_err))


        # SVM margin lines + support vectors 
        if draw_svm_margins:
            svm_margin_lines(ax, model, xx, yy, grid)

        # K-means punkter + labels (samma presentation)
        ax.scatter(X2[:, 0], X2[:, 1], c=kmlab, cmap="Accent", s=10, alpha=0.35, label="K-means kluster")
        ax.scatter(X2[y == 0, 0], X2[y == 0, 1], s=10, c="royalblue", alpha=0.75, label="Label 0")
        ax.scatter(X2[y == 1, 0], X2[y == 1, 1], s=20, c="crimson",   alpha=0.90, label="Label 1")

        ax.set_title(title)
        ax.set_xlabel("PC1")
        ax.grid(True, alpha=0.2)
        ax.legend(loc="best", frameon=False)

    axes[0].set_ylabel("PC2")
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[Sparat] {save_path}")
    if show:
        plt.show()

#%%
# =========================================================
#   Grade 5  2),  använda wrappers och embedded metoder
# =========================================================
def plot_side_by_side_compare_A(
    X2_list, y, kmlab_list, models, titles,
    save_path="svm_side_by_side_compare_A.png",
    show=True,
    draw_svm_margins=True
):
    """
    Tre paneler där varje panel har eget PCA-2D-rum (X2),
    eget KMeans-label (kmlab), och egen SVM (modell).
    """
    fig, axes = plt.subplots(1, 3, figsize=(19, 5))

    for i, ax in enumerate(axes):
        X2 = X2_list[i]
        kmlab = kmlab_list[i]
        model = models[i]
        title = titles[i]

        xx, yy, grid = make_meshgrid(X2, pad=0.6, step=0.02)
        Z = model.predict(grid).reshape(xx.shape)

        # bakgrund
        ax.contourf(xx, yy, Z, alpha=0.18, cmap="coolwarm")

        # boundary + margins + support vectors
        if draw_svm_margins:
            svm_margin_lines(ax, model, xx, yy, grid, draw_support_vectors=True)

        # punkter (nu blir svärmen olika pga olika PCA-rum)
        ax.scatter(X2[:, 0], X2[:, 1], c=kmlab, cmap="Accent", s=10, alpha=0.35, label="K-means kluster")
        ax.scatter(X2[y == 0, 0], X2[y == 0, 1], s=10, c="royalblue", alpha=0.75, label="Label 0")
        ax.scatter(X2[y == 1, 0], X2[y == 1, 1], s=20, c="crimson", alpha=0.90, label="Label 1")

        ax.set_title(title)
        ax.set_xlabel("PC1")
        ax.grid(True, alpha=0.2)
        ax.legend(loc="best", frameon=False)

    axes[0].set_ylabel("PC2")
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[Sparat] {save_path}")

    if show:
        plt.show()


#%%
# =========================================================
# 7) MAIN – använder stödfunktionerna (rätt flöde)
# =========================================================
def main():
    # 1) Läs data
    df = load_dataset("combined_dataset_normalized.csv")

    # 2) Features + labels
    X, y, feat_cols = get_X_y(df, label_col="label")

    # 3) PCA-2D
    pca, X2 = fit_pca_2d(X, seed=42)

    # 4) KMeans
    km, kmlab = fit_kmeans(X2, n_clusters=2, seed=42)

    # 5) Träna SVM med 80/20 (80_20-out) – använder split_80_20
    svm_80_20 = train_svm_80_20(X2, y, test_size=0.2, seed=42, C=1.0, gamma="scale")

    # 6) Träna SVM med 5-fold tuning – använder make_kfold
    svm_cv, best_params = train_svm_kfold(X2, y, k=5, seed=42)

    # 7) Plot sida-vid-sida – samma presentation
    plot_side_by_side(
        X2, y, kmlab, km, svm_80_20, svm_cv, best_params,
        save_path="svm_side_by_side.png",
        show=True,
        draw_svm_margins=True        # SVM boundary + margins + support vectors
    )


  



#%%
if __name__ == "__main__":
    main()