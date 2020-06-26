# Script based on blog post by Pubudu Samarakoon
import pandas as pd

path_gff3 = "../data/gencode_vM24_annotation.gff3"
path_output = "../output/"
file_name = "gene_locations_mouse_promoter_2k_and_gene.pkl"

gencode = pd.read_table(
    path_gff3,
    comment="#",
    sep="\t",
    names=[
        "seqname",
        "source",
        "feature",
        "start",
        "end",
        "score",
        "strand",
        "frame",
        "attribute",
    ],
)

gencode_genes = (
    gencode[(gencode.feature == "gene")][
        ["seqname", "start", "end", "attribute", "strand"]
    ]
    .copy()
    .reset_index()
    .drop("index", axis=1)
)  # Extract genes


def gene_info(x):
    # Extract gene names, gene_type, gene_status and level
    g_name = list(filter(lambda x: "gene_name" in x, x.split(";")))[0].split("=")[1]
    g_type = list(filter(lambda x: "gene_type" in x, x.split(";")))[0].split("=")[1]
    g_level = int(list(filter(lambda x: "level" in x, x.split(";")))[0].split("=")[1])
    return (g_name, g_type, g_level)


(
    gencode_genes["gene_name"],
    gencode_genes["gene_type"],
    gencode_genes["gene_level"],
) = zip(*gencode_genes.attribute.apply(lambda x: gene_info(x)))

gencode_genes["gene_type"].drop_duplicates()

# Sort gene_leve in each chromosome (ascending oder) and remove duplicates
gencode_genes = (
    gencode_genes.sort_values(["gene_level", "seqname"], ascending=True)
    .drop_duplicates("gene_name", keep="first")
    .reset_index()
    .drop("index", axis=1)
)

# Extend to promoter regions ONLY!!! upstream
extend_bases = 2000
gencode_genes.loc[gencode_genes["strand"] == "+", "start"] -= 2000
gencode_genes.loc[gencode_genes["strand"] == "-", "end"] += 2000

gencode_genes = gencode_genes.drop(["attribute", "gene_level", "strand"], axis=1)

gencode_genes.to_pickle(f"{path_output}{file_name}")
