import pandas as pd
import scanpy as sc
from sklearn.preprocessing import MinMaxScaler, RobustScaler

from src import prep
from src.model import Modeller

# Knowledge base RNA
path = "output/chen2019/Ad_BrainCortex/"
path_reference = "data/chen2019/Ad_BrainCortex/metadata.pkl"

# Read in RNA data
rna = pd.read_pickle(f"{path}RNAseq_prepped.pkl")
rna[rna.columns] = RobustScaler(quantile_range=(0.05, 0.95)).fit_transform(
    rna[rna.columns]
)
rna[rna.columns] = MinMaxScaler().fit_transform(rna[rna.columns])
prep.add_annotation(rna, path_reference)
print(f"Features in RNAseq data: {len(rna.columns)}")

# Knowledge base ATAC
path_mtx = "data/chen2019/Ad_BrainCortex/Ad_BrainCortex_SNAREseq_chromatin_counts.mtx"
path_genes = "data/chen2019/Ad_BrainCortex/Ad_BrainCortex_SNAREseq_chromatin_peaks.tsv"
path_barcodes = "data/chen2019/Ad_BrainCortex/Ad_BrainCortex_SNAREseq_chromatin_barcodes.tsv"
generate_gene_activity_matrix = False
use_gene_activity_matrix = True
needs_transpose = True
sc.settings.verbosity = 3

if generate_gene_activity_matrix or not use_gene_activity_matrix:
    atac = prep.atac(path=path, path_mtx=path_mtx, path_genes=path_genes, path_barcodes=path_barcodes)

if generate_gene_activity_matrix:
    prep.generate_gene_activity_matrix(atac, path)

if use_gene_activity_matrix:
    atac = pd.read_pickle(f"{path}ATACseq_ae_prepped_activity_matrix.pkl")

prep.add_annotation(atac, path_reference)
# atac = pd.read_pickle("output/chen2019/Ad_BrainCortex/ATACseq_ae_prepped.pkl")
# prep.add_annotation(atac, path_reference)
print(f"Features in ATACseq data: {len(atac.columns)}")

# Generate and train models
# Hyperparameters
tweak_cDNA_ae = {
    "act_encoder": "relu",
    "act_decoder": "relu",
    "act_restore": "sigmoid",
    "epochs": 300,
    "min_max_dim_comps": {"cDNA": [30, 3000]},
    "min_layers": 2,
    "max_layers": 6,
    "learning_rate": 0.01,
    "dropout_input": 0.5,
    "dropout_latent": 0.5,
    "loss": "binary_crossentropy",
}
tweak_chromatin_ae = {
    "act_encoder": "relu",
    "act_decoder": "relu",
    "act_restore": "sigmoid",
    "epochs": 300,
    "min_max_dim_comps": {"Chromatin": [30, 5000]},
    "min_layers": 2,
    "max_layers": 6,
    "dropout_input": 0.5,
    "dropout_latent": 0.5,
    "learning_rate": 0.001,
    "loss": "binary_crossentropy",
}
tweak_multi_ae = {
    "act_encoder": "relu",
    "act_decoder": "relu",
    "act_restore": "sigmoid",
    "epochs": 300,
    "min_max_dim_comps": {"cDNA": [30, 2500], "Chromatin": [30, 5000], "Integration": [30, 100]},
    "min_layers": 2,
    "max_layers": 3,
    "dropout_input": 0.2,
    "dropout_latent": 0.0,
    "learning_rate": 0.001,
    "loss": "binary_crossentropy",
}
tweak_cDNA_vae = {
    "act_encoder": "relu",
    "act_decoder": "relu",
    "act_restore": "sigmoid",
    "epochs": 300,
    "min_max_dim_comps": {"cDNA": [30, 3000]},
    "min_layers": 2,
    "max_layers": 6,
    "dropout_input": 0.5,
    "dropout_latent": 0.5,
    "learning_rate": 0.001,
    "loss": "vae_loss",
}
tweak_chromatin_vae = {
    "act_encoder": "relu",
    "act_decoder": "relu",
    "act_restore": "sigmoid",
    "epochs": 200,
    "min_max_dim_comps": {"Chromatin": [30, 5000]},
    "min_layers": 2,
    "max_layers": 6,
    "dropout_input": 0.5,
    "dropout_latent": 0.5,
    "learning_rate": 0.001,
    "loss": "vae_loss",
}
tweak_multi_vae = {
    "act_encoder": "relu",
    "act_decoder": "relu",
    "act_restore": "sigmoid",
    "epochs": 300,
    "min_max_dim_comps": {"cDNA": [30, 2500], "Chromatin": [30, 5000], "Integration": [30, 100]},
    "min_layers": 2,
    "max_layers": 3,
    "dropout_input": 0.2,
    "dropout_latent": 0.0,
    "learning_rate": 0.001,
    "loss": "vae_loss",
}

# TWEAKED @ 24.05.2020
hp_dict_cDNA_ae = {
    "act_encoder": "relu",
    "act_decoder": "relu",
    "act_restore": "sigmoid",
    "epochs": 300,
    "layer_dims": {"cDNA": [30, 476, 1195]},
    "learning_rate": 0.01,
    "dropout_input": 0.5,
    "dropout_latent": 0.5,
    "loss": "binary_crossentropy",
}
hp_dict_chromatin_ae = {
    "act_encoder": "relu",
    "act_decoder": "relu",
    "act_restore": "sigmoid",
    "epochs": 450,
    "layer_dims": {"Chromatin": [84, 1798, 5000]},
    "learning_rate": 0.001,
    "dropout_input": 0.5,
    "dropout_latent": 0.5,
    "loss": "binary_crossentropy",
}
hp_dict_multi_ae = {
    "act_encoder": "relu",
    "act_decoder": "relu",
    "act_restore": "sigmoid",
    "epochs": 500,
    "layer_dims": {"cDNA": [500, 1000], "Chromatin": [500, 1000]},
    "layer_dims_post_merge": [30, 300],
    # "layer_dims": {"cDNA": [30, 190, 1195], "Chromatin": [30, 1800, 5000]},
    # "layer_dims_post_merge": [40],
    "learning_rate": 0.001,
    "dropout_input": 0.2,
    "dropout_latent": 0.0,
    "loss": "binary_crossentropy",
}

hp_dict_cDNA_vae = {
    "act_encoder": "relu",
    "act_decoder": "relu",
    "act_restore": "sigmoid",
    "epochs": 300,
    "layer_dims": {"cDNA": [30, 190, 1195]},
    "learning_rate": 0.001,
    "dropout_input": 0.5,
    "dropout_latent": 0.5,
    "loss": "vae_loss",
}
hp_dict_chromatin_vae = {
    "act_encoder": "relu",
    "act_decoder": "relu",
    "act_restore": "sigmoid",
    "epochs": 450,
    "layer_dims": {"Chromatin": [104, 1249, 15001]},
    # "layer_dims": {"Chromatin": [1250, 4330]},
    "learning_rate": 0.001,
    "dropout_input": 0.5,
    "dropout_latent": 0.5,
    "loss": "vae_loss",
}
hp_dict_multi_vae = {
    "act_encoder": "relu",
    "act_decoder": "relu",
    "act_restore": "sigmoid",
    "epochs": 500,
    "layer_dims": {"cDNA": [500, 1000], "Chromatin": [500, 1000]},
    "layer_dims_post_merge": [30, 300],
    # "layer_dims": {"cDNA": [190, 1195], "Chromatin": [1250, 4330]},
    # "layer_dims_post_merge": [300, 1000],
    # "layer_dims": {"cDNA": [100, 300], "Chromatin": [140, 670]},
    # "layer_dims_post_merge": [60, 150],
    "learning_rate": 0.001,
    "dropout_input": 0.2,
    "dropout_latent": 0.0,
    "loss": "vae_loss",
}
model = Modeller(output_path="dataset_scripts/Ad_BrainCortex/latex_final/")

# Add components to model
model.add_component("cDNA", comp=rna)
model.add_component("Chromatin", comp=atac)

# Baseline UMAPS
model.plot_umap(name="cDNA", type="Raw data", n_neighbors=50, resolution=1, tweak_resolution=True, k_classes=[6,10], raw=True)
model.plot_umap(name="Chromatin", type="Raw data", n_neighbors=50, resolution=1, tweak_resolution=True, k_classes=[6,10], raw=True)
model.plot_umap(name="Multiomics_Integration", type="Raw data", n_neighbors=50, resolution=1, tweak_resolution=True, k_classes=[6,10], raw=True)

# Fit [V]AEs
model.fit(type="AE", name="cDNA", hp=hp_dict_cDNA_ae)
model.fit(type="AE", name="Chromatin", hp=hp_dict_chromatin_ae)
model.fit(type="AE", name="Multiomics_Integration", hp=hp_dict_multi_ae)
model.fit(type="VAE", name="cDNA", hp=hp_dict_cDNA_vae)
model.fit(type="VAE", name="Chromatin", hp=hp_dict_chromatin_vae)
model.fit(type="VAE", name="Multiomics_Integration", hp=hp_dict_multi_vae)

# Plot umaps and evaluate results
model.plot_umap(type="AE", name="cDNA", n_neighbors=50, resolution=0.5, tweak_resolution=True, k_classes=[6,10])
# model.plot_umap(type="AE", name="cDNA", n_neighbors=50, resolution=0.003689348814741915)
model.plot_umap(type="AE", name="Chromatin", n_neighbors=50, resolution=0.5, tweak_resolution=True, k_classes=[6,10])
model.plot_umap(type="AE", name="Multiomics_Integration", n_neighbors=50, resolution=0.5, tweak_resolution=True, k_classes=[6,10])
model.plot_umap(type="VAE", name="cDNA", n_neighbors=50, resolution=0.5, tweak_resolution=True, k_classes=[6,10])
model.plot_umap(type="VAE", name="Chromatin", n_neighbors=50, resolution=0.5, tweak_resolution=True, k_classes=[6,10])
model.plot_umap(type="VAE", name="Multiomics_Integration", n_neighbors=50, resolution=0.5, tweak_resolution=True, k_classes=[6,10])

model.evaluate()

# Label Transfer
model.add_train_test(test_split=0.5)
gene_map = pd.read_pickle("output/gene_locations_mouse_promoter_2k_and_gene.pkl")
gene_map = gene_map[gene_map["gene_type"] == "protein_coding"]

model.transfer_label(type="AE", name="cDNA", k_classes=10, hp=hp_dict_cDNA_ae, gene_loc_df=gene_map, explain=False, lr=0.001)
model.transfer_label(type="AE", name="Chromatin", k_classes=10, hp=hp_dict_chromatin_ae, gene_loc_df=gene_map, explain=False, lr=0.0001, epochs=400)
model.transfer_label(type="AE", name="Multiomics_Integration", k_classes=10, hp=hp_dict_multi_ae, gene_loc_df=gene_map, explain=False, lr=0.0001, epochs=400)
model.transfer_label(type="VAE", name="cDNA", k_classes=10, hp=hp_dict_cDNA_vae, gene_loc_df=gene_map, explain=False, lr=0.001)
model.transfer_label(type="VAE", name="Chromatin", k_classes=10, hp=hp_dict_chromatin_vae, gene_loc_df=gene_map, explain=False, lr=0.0001, epochs=400)
model.transfer_label(type="VAE", name="Multiomics_Integration", k_classes=10, hp=hp_dict_multi_vae, gene_loc_df=gene_map, explain=False, lr=0.0001, epochs=400)

model.evaluate()

# Tweak hyperparameters
# model.auto_tweak(name="cDNA", type="AE", tweak_hp=tweak_cDNA_ae, n_neighbors=50, resolution=0.4, k_classes=6)
# model.auto_tweak(name="Chromatin", type="AE", tweak_hp=tweak_chromatin_ae, n_neighbors=50, resolution=0.4, k_classes=6)
# model.auto_tweak(name="Multiomics_Integration", type="AE", tweak_hp=tweak_multi_ae, n_neighbors=50, resolution=0.4, k_classes=6)
# model.auto_tweak(name="cDNA", type="VAE", tweak_hp=tweak_cDNA_vae, n_neighbors=50, resolution=0.4, k_classes=6)
# model.auto_tweak(name="Chromatin", type="VAE", tweak_hp=tweak_chromatin_vae, n_neighbors=50, resolution=0.4, k_classes=6)
# model.auto_tweak(name="Multiomics_Integration", type="VAE", tweak_hp=tweak_multi_vae, n_neighbors=50, resolution=0.4, k_classes=6)

# 06.06.2020
#                                  Adjusted mutual information score  Silhouette Score  Label transfer accuracy
# cDNA_Raw data                                             0.000416         -0.087238                      NaN
# Chromatin_Raw data                                        0.001464         -0.302249                      NaN
# Multiomics_Integration_Raw data                           0.000703         -0.266051                      NaN
# cDNA_AE                                                   0.614929         -0.240252                   0.8148
# Chromatin_AE                                              0.335398         -0.326795                   0.0249
# Multiomics_Integration_AE                                 0.439727         -0.268684                   0.8181
# cDNA_VAE                                                  0.444271         -0.267997                   0.8136
# Chromatin_VAE                                             0.000016         -0.183879                   0.0379
# Multiomics_Integration_VAE                                0.343632         -0.270852                   0.8124

#                                  Adjusted mutual information score  Silhouette Score  Label transfer accuracy
# cDNA_Raw data                                            -0.000088         -0.092952                      NaN
# Chromatin_Raw data                                        0.000365         -0.217620                      NaN
# Multiomics_Integration_Raw data                           0.001360         -0.184668                      NaN
# cDNA_AE                                                   0.352090         -0.158585                   0.8214
# Chromatin_AE                                              0.340401         -0.299942                   0.6982
# Multiomics_Integration_AE                                 0.461986         -0.277075                   0.8033
# cDNA_VAE                                                  0.380577         -0.242304                   0.8128
# Chromatin_VAE                                             0.290156         -0.272418                   0.6981
# Multiomics_Integration_VAE                                0.324819         -0.291950                   0.7916


