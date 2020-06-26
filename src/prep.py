import itertools

import pandas as pd
import progressbar
import scanpy as sc
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.preprocessing import MaxAbsScaler


def add_annotation(df, path_reference):
    comparison_clusters = pd.read_pickle(path_reference)[["Barcode", "IdentChar"]]
    comparison_clusters = comparison_clusters.set_index("Barcode")
    idx_both = df.index.intersection(comparison_clusters.index)
    comparison_clusters = comparison_clusters.loc[idx_both]
    df["Annotation"] = comparison_clusters


def atac(path, ad=False, path_mtx=None, path_genes=None, path_barcodes=None):
    # Import and transpose data
    if ad is True:
        df = sc.read(path_mtx).transpose()
        df = pd.DataFrame(data=df.X, index=df.obs_names, columns=df.var_names)
    else:
        df = sc.read_mtx(path_mtx).transpose()
        barcodes = pd.read_csv(path_barcodes, delimiter="\t", header=None)
        df.obs_names = barcodes[0].values
        genes = pd.read_csv(path_genes, delimiter="\t", header=None)
        df.var_names = genes[0].values
        df = pd.DataFrame(data=df.X.toarray(), index=df.obs_names, columns=df.var_names)

    # Use only cells not discarded by RNAseq filtering
    inds = pd.read_pickle(f"{path}RNAseq_prepped.pkl").index
    df = df.loc[inds]

    # Normalize data
    df[df.columns] = MaxAbsScaler().fit_transform(df[df.columns])

    # Term frequency - Inverse document frequency
    tf_idf = TfidfTransformer(norm=None, sublinear_tf=False)
    transformed = tf_idf.fit_transform(df.values).toarray()
    transformed_scaled = MaxAbsScaler().fit_transform(transformed)
    df = pd.DataFrame(transformed_scaled, index=df.index, columns=df.columns)

    # Select top peaks
    df = df.loc[
         :, df.columns[(df != 0).sum() >= (df[df.columns] != 0).sum().quantile(0.75)]
         ]
    col_vars = df.var(axis=0)
    highly_variable_cols = df.columns[col_vars >= col_vars.quantile(0.75)]
    df = df.loc[:, highly_variable_cols]
    df.to_pickle(f"{path}ATACseq_ae_prepped.pkl")

    return df


def get_genes(df, loc):
    # Define region
    chr = loc.split(":")[0]
    start = int(loc.split(":")[1].split("-")[0])
    end = int(loc.split(":")[1].split("-")[1])

    # Find partial and complete overlaps
    df["diff_ss"] = ((df[df["seqname"] == chr]["start"] - 2000) - start) <= 0
    df["diff_es"] = (df[df["seqname"] == chr]["end"] - start) >= 0
    df["diff_se"] = ((df[df["seqname"] == chr]["start"] - 2000) - end) <= 0
    df["diff_ee"] = (df[df["seqname"] == chr]["end"] - end) >= 0
    df.fillna(value=False, inplace=True)

    df["overlap"] = (
        (df["diff_ss"] & df["diff_es"])
        | (df["diff_ee"] & df["diff_se"])
        | (df["diff_ss"] & df["diff_es"] & df["diff_ee"] & df["diff_se"])
    )

    # Return list of genes overlapping queried region
    genes = df[df["overlap"]]["gene_name"].to_list()

    return genes


def generate_gene_activity_matrix(atac, path):
    print("Generating gene activity matrix...")
    genes_per_col = []
    bar = progressbar.ProgressBar()
    df = pd.read_pickle("output/gene_locations_mouse_promoter_2k_and_gene.pkl")
    df = df[df["gene_type"] == "protein_coding"]
    for c in bar(atac.columns):
        genes_per_col.append(get_genes(df, c))

    gene_atac_map = {
        gene: [] for gene in list(set(itertools.chain.from_iterable(genes_per_col)))
    }

    for gene in gene_atac_map:
        for frag_genes in genes_per_col:
            if gene in frag_genes:
                gene_atac_map[gene].append(genes_per_col.index(frag_genes))

    activity_matrix = pd.DataFrame()
    for gene in gene_atac_map:
        activity_matrix[f"{gene}_ATAC"] = atac.iloc[:, gene_atac_map[gene]].sum(axis=1)

    transformed_scaled = MaxAbsScaler().fit_transform(activity_matrix)
    activity_matrix = pd.DataFrame(transformed_scaled, index=activity_matrix.index, columns=activity_matrix.columns)

    activity_matrix.to_pickle(f"{path}ATACseq_ae_prepped_activity_matrix.pkl")
