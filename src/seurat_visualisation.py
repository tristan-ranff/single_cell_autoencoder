import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
from sklearn import metrics

from src import prep

path = "output/chen2019/Ad0.2827_BrainCortex/seurat/"
path_output = "dataset_scripts/Ad0.2827_BrainCortex/seurat_figures/"
path_reference = "data/chen2019/Ad0.2827_BrainCortex/metadata.pkl"

eval = pd.DataFrame(columns=["Method", "AMIS", "F1"])

for set in ["rna", "atac", "multi"]:

    embedding = pd.read_csv(f"{path}{set}_embedding.csv", index_col=0)
    #embedding.index = embedding.index.str.split("_").str[0]
    df_plt = pd.DataFrame(embedding.values, columns=["UMAP 1", "UMAP 2"])
    if set == "multi":
        embedding.index = embedding.index.str[:-2]
    prep.add_annotation(embedding, path_reference)
    df_plt["Cell type"] = embedding["Annotation"].to_list()
    df_plt["Seurat clusters"] = pd.read_csv(f"{path}{set}_clusters.csv", index_col=0)["x"].to_list()
    df_plt = df_plt.sort_values("Cell type")

    amis = metrics.adjusted_mutual_info_score(
        labels_true=pd.factorize(df_plt["Cell type"])[0],
        labels_pred=df_plt["Seurat clusters"].to_list(),
    )
    print(f"Cluster similarity: {amis:.4f}")


    fig, ax = plt.subplots(ncols=2, tight_layout=True, figsize=(12, 4))

    sns.scatterplot(
        x="UMAP 1",
        y="UMAP 2",
        hue="Seurat clusters",
        data=df_plt,
        s=4,
        linewidth=0,
        palette="Set2",
        ax=ax[0],
    )
    ax[0].legend(
        loc="center left", bbox_to_anchor=(1.05, 0.5), ncol=1, prop={"size": 8}
    )
    ax[0].set_anchor("W")
    ax[0].set_title(f"Seurat {set} clustering")

    sns.scatterplot(
        x="UMAP 1",
        y="UMAP 2",
        hue="Cell type",
        data=df_plt,
        s=4,
        linewidth=0,
        palette="Set2",
        ax=ax[1],
    )
    ax[1].legend(
        loc="center left", bbox_to_anchor=(1.05, 0.5), ncol=1, prop={"size": 8}
    )
    ax[1].set_title(f"Seurat {set} reference annotation")
    plt.savefig(
        f"{path_output}{set}_combined_umap.png", dpi=300,
    )
    plt.show()
    plt.close()

    if "multi" in set:
        eval.loc[len(eval), :] = [set, amis, np.nan]
        continue

    predictions = pd.read_csv(f"{path}{set}_predictions.csv", index_col=0)
    cm = pd.crosstab(predictions.iloc[:, 1], predictions.iloc[:, 0])
    for ind in cm.index:
        if ind in cm.columns:
            continue
        else:
            cm[ind] = 0
    cm = cm.reindex(columns=cm.index)
    cm.loc[:, len(cm.columns)] = 0
    cm.loc[len(cm), :] = 0
    cm.loc["Total"] = cm.sum()
    cm.loc[:, "Total"] = cm.sum(axis=1)
    mask = np.zeros_like(cm, dtype=np.bool)
    mask[:, -2] = True
    mask[-2, :] = True
    plt.subplots(figsize=(10, 10))
    ax = sns.heatmap(
        cm.astype(int),
        mask=mask,
        linewidths=0.25,
        cmap="Blues",
        cbar=False,
        annot=True,
        fmt="d",
    )
    ax.xaxis.set_ticks_position("top")
    ax.xaxis.set_label_position("top")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
    xticks = ax.xaxis.get_major_ticks()
    xticks[-2].set_visible(False)
    yticks = ax.yaxis.get_major_ticks()
    yticks[-2].set_visible(False)

    for i in range(0, predictions.iloc[:,1].nunique()):
        ax.add_patch(Rectangle((i, i), 1, 1, fill=False, edgecolor="black", lw=2))

    plt.xlabel("Predicted labels", fontsize=16)
    plt.ylabel("Reference labels", fontsize=16)
    plt.title(f"Contingency matrix Seurat {set}_test", fontsize=20)
    plt.savefig(
        f"{path_output}seurat_{set}_test_contingency_matrix.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.show()
    plt.close()

    precision = metrics.classification_report(
        y_true=predictions.iloc[:,1], y_pred=predictions.iloc[:,0], zero_division=0, output_dict=True
    )["weighted avg"]["f1-score"]
    print(f"Cluster similarity: {precision:.4f}")
    eval.loc[len(eval), :] = [set, amis, precision]

eval.to_csv(f"{path_output}evaluation.csv")