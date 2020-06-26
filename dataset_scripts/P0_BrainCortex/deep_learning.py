import pandas as pd
import scanpy as sc
from sklearn.preprocessing import MinMaxScaler, RobustScaler

from src import prep
from src.model import Modeller

# Knowledge base RNA
path = "output/chen2019/P0_BrainCortex/"
path_reference = "data/chen2019/P0_BrainCortex/metadata.pkl"

# Read in RNA data
rna = pd.read_pickle(f"{path}RNAseq_prepped.pkl")
rna[rna.columns] = RobustScaler(quantile_range=(0.05, 0.95)).fit_transform(
    rna[rna.columns]
)
rna[rna.columns] = MinMaxScaler().fit_transform(rna[rna.columns])
prep.add_annotation(rna, path_reference)
print(f"Features in RNAseq data: {len(rna.columns)}")

# Knowledge base ATAC
path_mtx = "data/chen2019/P0_BrainCortex/GSE126074_P0_BrainCortex_SNAREseq_chromatin_counts.mtx"
path_genes = "data/chen2019/P0_BrainCortex/GSE126074_P0_BrainCortex_SNAREseq_chromatin_peaks.tsv"
path_barcodes = "data/chen2019/P0_BrainCortex/GSE126074_P0_BrainCortex_SNAREseq_chromatin_barcodes.tsv"
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
# atac = pd.read_pickle("output/chen2019/P0_BrainCortex/ATACseq_ae_prepped.pkl")
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
    "min_max_dim_comps": {"Chromatin": [30, 10000]},
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
    "epochs": 300,
    "min_max_dim_comps": {"Chromatin": [30, 10000]},
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

# TWEAKED @ 24.05.2020 CHECKED
hp_dict_cDNA_ae = {
    "act_encoder": "relu",
    "act_decoder": "relu",
    "act_restore": "sigmoid",
    "epochs": 300,
    "layer_dims": {"cDNA": [30, 190, 476]},
    "learning_rate": 0.01,
    "dropout_input": 0.5,
    "dropout_latent": 0.5,
    "loss": "binary_crossentropy",
}
# TWEAKED @ 24.05.2020 CHECKED
hp_dict_chromatin_ae = {
    "act_encoder": "relu",
    "act_decoder": "relu",
    "act_restore": "sigmoid",
    "epochs": 400,
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
    "epochs": 400,
    "layer_dims": {"cDNA": [30, 300], "Chromatin": [388, 5000]},
    "layer_dims_post_merge": [50, 100],
    # "layer_dims": {"cDNA": [30, 190, 1195], "Chromatin": [30, 1800, 5000]},
    # "layer_dims_post_merge": [40],
    "learning_rate": 0.001,
    "dropout_input": 0.2,
    "dropout_latent": 0.0,
    "loss": "binary_crossentropy",
}

# TWEAKED @ 24.05.2020 CHECKED
hp_dict_cDNA_vae = {
    "act_encoder": "relu",
    "act_decoder": "relu",
    "act_restore": "sigmoid",
    "epochs": 300,
    "layer_dims": {"cDNA": [190, 476, 1195, 3001]},
    "learning_rate": 0.001,
    "dropout_input": 0.5,
    "dropout_latent": 0.5,
    "loss": "vae_loss",
}

# TWEAKED @ 23.05.20 CHECKED
hp_dict_chromatin_vae = {
    "act_encoder": "relu",
    "act_decoder": "relu",
    "act_restore": "sigmoid",
    "epochs": 400,
    "layer_dims": {"Chromatin": [104, 1249, 15001]},
    "learning_rate": 0.001,
    "dropout_input": 0.5,
    "dropout_latent": 0.5,
    "loss": "vae_loss",
}
hp_dict_multi_vae = {
    "act_encoder": "relu",
    "act_decoder": "relu",
    "act_restore": "sigmoid",
    "epochs": 400,
    "layer_dims": {"cDNA": [30, 2500], "Chromatin": [388, 5000]},
    "layer_dims_post_merge": [55, 100],
    # "layer_dims": {"cDNA": [190, 1195], "Chromatin": [1250, 4330]},
    # "layer_dims_post_merge": [300, 1000],
    # "layer_dims": {"cDNA": [100, 300], "Chromatin": [140, 670]},
    # "layer_dims_post_merge": [60, 150],
    "learning_rate": 0.001,
    "dropout_input": 0.2,
    "dropout_latent": 0.0,
    "loss": "vae_loss",
}
model = Modeller(output_path="dataset_scripts/P0_BrainCortex/latex_final/")

# Add components to model
model.add_component("cDNA", comp=rna)
model.add_component("Chromatin", comp=atac)

# Baseline UMAPS
# model.plot_umap(name="cDNA", type="Raw data", n_neighbors=50, resolution=1, tweak_resolution=True, k_classes=[6,9], raw=True)
# model.plot_umap(name="Chromatin", type="Raw data", n_neighbors=50, resolution=1, tweak_resolution=True, k_classes=[6,9], raw=True)
# model.plot_umap(name="Multiomics_Integration", type="Raw data", n_neighbors=50, resolution=1, tweak_resolution=True, k_classes=[6,9], raw=True)
# #
# # # Fit [V]AEs
# model.fit(type="AE", name="cDNA", hp=hp_dict_cDNA_ae)
# model.fit(type="AE", name="Chromatin", hp=hp_dict_chromatin_ae)
# model.fit(type="AE", name="Multiomics_Integration", hp=hp_dict_multi_ae)
# model.fit(type="VAE", name="cDNA", hp=hp_dict_cDNA_vae)
# model.fit(type="VAE", name="Chromatin", hp=hp_dict_chromatin_vae)
# model.fit(type="VAE", name="Multiomics_Integration", hp=hp_dict_multi_vae)
# #
# # # Plot umaps and evaluate results
# model.plot_umap(type="AE", name="cDNA", n_neighbors=50, resolution=0.5, tweak_resolution=True, k_classes=[6,9])
# model.plot_umap(type="AE", name="Chromatin", n_neighbors=50, resolution=0.5, tweak_resolution=True, k_classes=[6,9])
# model.plot_umap(type="AE", name="Multiomics_Integration", n_neighbors=50, resolution=0.5, tweak_resolution=True, k_classes=[6,9])
# model.plot_umap(type="VAE", name="cDNA", n_neighbors=50, resolution=0.5, tweak_resolution=True, k_classes=[6,9])
# model.plot_umap(type="VAE", name="Chromatin", n_neighbors=50, resolution=0.5, tweak_resolution=True, k_classes=[6,9])
# model.plot_umap(type="VAE", name="Multiomics_Integration", n_neighbors=50, resolution=0.5, tweak_resolution=True, k_classes=[6,9])
# model.plot_umap(type="AE", name="cDNA", n_neighbors=50, resolution=0.017592186044416015)
# model.plot_umap(type="AE", name="Chromatin", n_neighbors=50, resolution=0.10485760000000005)
# model.plot_umap(type="AE", name="Multiomics_Integration", n_neighbors=50, resolution=0.06710886400000003)
# model.plot_umap(type="VAE", name="cDNA", n_neighbors=50, resolution=0.08388608000000004)
# model.plot_umap(type="VAE", name="Chromatin", n_neighbors=50, resolution=0.08388608000000004)
# model.plot_umap(type="VAE", name="Multiomics_Integration", n_neighbors=50, resolution=0.08388608000000004)

model.evaluate()

# Label Transfer
model.add_train_test(test_split=0.5)
gene_map = pd.read_pickle("output/gene_locations_mouse_promoter_2k_and_gene.pkl")
gene_map = gene_map[gene_map["gene_type"] == "protein_coding"]

# model.transfer_label(type="AE", name="cDNA", k_classes=9, hp=hp_dict_cDNA_ae, gene_loc_df=gene_map, explain=False, lr=0.001)
# model.transfer_label(type="AE", name="Chromatin", k_classes=9, hp=hp_dict_chromatin_ae, gene_loc_df=gene_map, explain=False, lr=0.0001, epochs=400)
# model.transfer_label(type="AE", name="Multiomics_Integration", k_classes=9, hp=hp_dict_multi_ae, gene_loc_df=gene_map, explain=False, lr=0.0001, epochs=400)
# model.transfer_label(type="VAE", name="cDNA", k_classes=9, hp=hp_dict_cDNA_vae, gene_loc_df=gene_map, explain=False, lr=0.001)
# model.transfer_label(type="VAE", name="Chromatin", k_classes=9, hp=hp_dict_chromatin_vae, gene_loc_df=gene_map, explain=False, lr=0.0001, epochs=400)
# model.transfer_label(type="VAE", name="Multiomics_Integration", k_classes=9, hp=hp_dict_multi_vae, gene_loc_df=gene_map, explain=False, lr=0.0001, epochs=400)


# Tweak hyperparameters
# model.auto_tweak(name="cDNA", type="AE", tweak_hp=tweak_cDNA_ae, n_neighbors=50, resolution=0.4, k_classes=6)
# model.auto_tweak(name="Chromatin", type="AE", tweak_hp=tweak_chromatin_ae, n_neighbors=50, resolution=0.4, k_classes=6)
# model.auto_tweak(name="Multiomics_Integration", type="AE", tweak_hp=tweak_multi_ae, n_neighbors=50, resolution=0.4, k_classes=6)
# model.auto_tweak(name="cDNA", type="VAE", tweak_hp=tweak_cDNA_vae, n_neighbors=50, resolution=0.4, k_classes=6)
# model.auto_tweak(name="Chromatin", type="VAE", tweak_hp=tweak_chromatin_vae, n_neighbors=50, resolution=0.4, k_classes=6)
# model.auto_tweak(name="Multiomics_Integration", type="VAE", tweak_hp=tweak_multi_vae, n_neighbors=50, resolution=0.4, k_classes=6)

model.evaluate()

# 06.06.2020

#                                  Adjusted mutual information score  Silhouette Score  Label transfer accuracy
# cDNA_Raw data                                             0.004187         -0.038087                      NaN
# Chromatin_Raw data                                        0.001706         -0.178174                      NaN
# Multiomics_Integration_Raw data                           0.001494         -0.151050                      NaN
# cDNA_AE                                                   0.410732         -0.102976                   0.7945
# Chromatin_AE                                              0.503649         -0.151090                   0.0004
# Multiomics_Integration_AE                                 0.522989         -0.210543                   0.7506
# cDNA_VAE                                                  0.495601         -0.106947                   0.7759
# Chromatin_VAE                                             0.450335         -0.250828                   0.4828
# Multiomics_Integration_VAE                                0.500986         -0.224714                   0.6270

# 06.06.2020
#                                  Adjusted mutual information score  Silhouette Score  Label transfer accuracy
# cDNA_Raw data                                             0.000231         -0.055440                      NaN
# Chromatin_Raw data                                       -0.000446         -0.290743                      NaN
# Multiomics_Integration_Raw data                           0.001076         -0.228310                      NaN
# cDNA_AE                                                   0.412579         -0.137073                   0.8132
# Chromatin_AE                                              0.520941         -0.157801                   0.6363
# Multiomics_Integration_AE                                 0.531820         -0.158302                   0.6399
# cDNA_VAE                                                  0.436900         -0.098326                   0.7984
# Chromatin_VAE                                             0.428624         -0.243685                   0.6546
# Multiomics_Integration_VAE                                0.413543         -0.244040                   0.6178

