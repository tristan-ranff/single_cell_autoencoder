import itertools
import os
import time

import anndata
import lime
import lime.lime_tabular
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import progressbar
import regex as re
import scanpy as sc
import seaborn as sns
import tensorflow as tf
import umap
from keras import backend as K
from keras.callbacks import EarlyStopping
from keras.layers import Concatenate, Dense, Dropout, Input, Lambda
from keras.losses import binary_crossentropy
from keras.models import Model
from keras.optimizers import Adam
from keras.utils import plot_model
from matplotlib.patches import Rectangle
from sklearn import metrics
from sklearn.model_selection import train_test_split

from src import heatmap_helper


#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


def undersampler(df_comps):
    while True:
        x = []
        y = []
        for df in df_comps:
            batch = pd.DataFrame()
            min = df.groupby("Annotation").size().min()
            for name, grp in df.groupby("Annotation"):
                undersample = grp.sample(min)
                batch = pd.concat([batch, undersample])
            batch = batch.drop(columns="Annotation")
            batch = batch.sample(frac=1).values
            x.append(batch)
            y.append(batch)
        yield [x, y]


def fit_comps(m, name, type, comp_dfs, hp, output, n_centroids, verbose, es=False, build_only=False, beta=1):
    multi = "Multiomics_Integration" in name
    use_train = ""
    if "_train" in name:
        use_train = True
    start = 1
    if verbose == 0:
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

    # VAE functions
    def sampling(args):
        z_mean, z_log_var = args
        batch = K.shape(z_mean)[0]
        dim = K.int_shape(z_mean)[1]
        epsilon = K.random_normal(shape=(batch, dim))
        return z_mean + K.exp(0.5 * z_log_var) * epsilon

    def vae_loss(y_true, y_pred):
        reconstruction_loss = (
            data.shape[0]
            * data.shape[1]
            * binary_crossentropy(K.flatten(y_true), K.flatten(y_pred))
        )
        kl_loss = -0.5 * K.sum(
            1
            - K.exp(model.get_layer("Variance").output)
            - K.square(model.get_layer("Mean").output)
            + model.get_layer("Variance").output,
            axis=1,
        )
        return reconstruction_loss + beta * kl_loss

    if not multi:
        comp_dfs = {f"{name}": comp_dfs[f"{name}"]}

    else:
        if use_train:
            comp_dfs = {
                f"{n}_train": comp_dfs[f"{n}_train"]
                for n in comp_dfs.keys()
                if "_train" not in n and "_test" not in n
            }
        else:
            comp_dfs = {
                f"{n}": comp_dfs[f"{n}"]
                for n in comp_dfs.keys()
                if "_train" not in n and "_test" not in n
            }

    inputs = []
    last_layers = []
    for key, value in comp_dfs.items():
        if use_train:
            key = key.split("_train")[0]
        data = value.iloc[:, :-1].values
        input = Input(shape=(data.shape[1],), name=f"Input_{key}")
        dropout = Dropout(rate=hp["dropout_input"], name=f"Dropout_{key}_input")(input)
        for i, d in enumerate(reversed(hp["layer_dims"][key])):
            if i == 0:  # and i != len(hp["layer_dims"][key]) - 1:
                new_dense_enc = Dense(
                    d, activation=hp["act_encoder"], name=f"{key}_encoder_{i + 1}"
                )(dropout)
            elif i == len(hp["layer_dims"][key]) - 1 and not multi:
                dropout_lat = Dropout(rate=0.5, name="Dropout_latent")(new_dense_enc)
                new_dense_enc = Dense(
                    d, activation=hp["act_encoder"], name=f"Bottleneck"
                )(dropout_lat)
                if type == "VAE":
                    # Latent distribution and sampling
                    mu = Dense(d, name="Mean")(new_dense_enc)
                    sigma = Dense(d, name="Variance")(new_dense_enc)
                    z = Lambda(sampling, output_shape=(d,), name="Sampler")([mu, sigma])
            else:
                new_dense_enc = Dense(
                    d, activation=hp["act_encoder"], name=f"{key}_encoder_{i + 1}"
                )(new_dense_enc)
        last_layers.append(new_dense_enc)
        inputs.append(input)

    if multi:
        concat = Concatenate(name="Concatenate")(last_layers)
        for i, d in enumerate(reversed(hp["layer_dims_post_merge"])):
            if i == 0 and i != len(hp["layer_dims_post_merge"]) - 1:
                new_dense_menc = Dense(
                    d, activation=hp["act_encoder"], name=f"Merged_encoder_{i + 1}"
                )(concat)
            elif i == len(hp["layer_dims_post_merge"]) - 1:
                if i == 0:
                    dropout_lat = Dropout(rate=hp["dropout_latent"], name="Dropout_latent")(concat)
                else:
                    dropout_lat = Dropout(rate=hp["dropout_latent"], name="Dropout_latent")(new_dense_menc)
                new_dense_menc = Dense(
                    d, activation=hp["act_encoder"], name=f"Bottleneck"
                )(dropout_lat)
                if type == "VAE":
                    # Latent distribution and sampling
                    mu = Dense(d, name="Mean")(new_dense_menc)
                    sigma = Dense(d, name="Variance")(new_dense_menc)
                    z = Lambda(sampling, output_shape=(d,), name="Sampler")([mu, sigma])
            else:
                new_dense_menc = Dense(
                    d, activation=hp["act_encoder"], name=f"Merged_encoder_{i+1}"
                )(new_dense_menc)

        more_than_one = len(hp["layer_dims_post_merge"]) > 1
        if more_than_one:
            for i, d in enumerate(hp["layer_dims_post_merge"][1:]):
                if i == 0:
                    if type == "AE":
                        new_dense_mdenc = Dense(
                            d,
                            activation=hp["act_decoder"],
                            name=f"Merged_decoder_{i + 1}",
                        )(new_dense_menc)
                    elif type == "VAE":
                        new_dense_mdenc = Dense(
                            d,
                            activation=hp["act_decoder"],
                            name=f"Merged_decoder_{i + 1}",
                        )(z)
                else:
                    new_dense_mdenc = Dense(
                        d, activation=hp["act_decoder"], name=f"Merged_decoder_{i+1}"
                    )(new_dense_mdenc)

        # Split components
        split_dim = 0
        for value in hp["layer_dims"].values():
            split_dim += min(value)
        if more_than_one:
            split = Dense(split_dim, activation=hp["act_decoder"], name="Split")(
                new_dense_mdenc
            )
        else:
            if type == "VAE":
                split = Dense(split_dim, activation=hp["act_decoder"], name="Split")(z)
            else:
                split = Dense(split_dim, activation=hp["act_decoder"], name="Split")(
                    new_dense_menc
                )
        start = 0

    outputs = []
    for key, value in comp_dfs.items():
        if use_train:
            key = key.split("_train")[0]
        data = value.iloc[:, :-1].values
        if start >= len(hp["layer_dims"][key]):
            continue
        for i, d in enumerate(hp["layer_dims"][key][start:]):
            if i == 0:
                if multi:
                    new_dense_denc = Dense(
                        d, activation=hp["act_decoder"], name=f"{key}_decoder_{i + 1}"
                    )(split)
                else:
                    if type == "AE":
                        new_dense_denc = Dense(
                            d,
                            activation=hp["act_decoder"],
                            name=f"{key}_decoder_{i + 1}",
                        )(new_dense_enc)
                    elif type == "VAE":
                        new_dense_denc = Dense(
                            d,
                            activation=hp["act_decoder"],
                            name=f"{key}_decoder_{i + 1}",
                        )(z)
            else:
                new_dense_denc = Dense(
                    d, activation=hp["act_decoder"], name=f"{key}_decoder_{i+1}"
                )(new_dense_denc)

        # Restore input dim
        alt_data = Dense(
            data.shape[1], activation=hp["act_restore"], name=f"Output_{key}"
        )(new_dense_denc)
        outputs.append(alt_data)

    model = Model(inputs=inputs, outputs=outputs, name=f"{type}_{name}")

    # Summarize AE model and save model visualization plot
    if verbose == 1:
        model.summary()
    plot_model(
        model,
        dpi=300,
        expand_nested=True,
        show_shapes=True,
        to_file=f"{output}/{type}_{name}_model_architecture.png",
    )

    # Compile AE
    loss = hp["loss"]
    if loss == "vae_loss":
        loss = vae_loss
    model.compile(optimizer=Adam(learning_rate=hp["learning_rate"]), loss=loss)

    if es is True:
        es = [
            EarlyStopping(
                monitor="loss",
                min_delta=0,
                patience=30,
                verbose=1,
                mode="auto",
                restore_best_weights=True,
            )
        ]
    else:
        es = []

    if not build_only:
        estimator = model.fit_generator(
            undersampler(list(comp_dfs.values())),
            epochs=hp["epochs"],
            # steps_per_epoch=math.ceil(len(list(comp_dfs.values())[0]) / bs),
            steps_per_epoch=10,
            shuffle=True,
            verbose=verbose,
            callbacks=es,
        )

        # Plot model loss
        df_plot = pd.DataFrame(estimator.history)
        df_plot["Epoch"] = np.arange(0, len(df_plot))
        df_plot = df_plot.melt("Epoch", var_name="Components", value_name="Loss")
        g = sns.lineplot(x="Epoch", y="Loss", hue="Components", data=df_plot)
        g.figure.savefig(f"{output}/{type}_{name}_training.png")
        plt.close()

    # Drop decoder and get bottleneck representation for data
    if type == "VAE":
        encoder = Model(inputs=model.input, outputs=model.get_layer("Sampler").output)
    else:
        encoder = Model(
            inputs=model.input, outputs=model.get_layer("Bottleneck").output
        )
    data = [df.iloc[:, :-1].values for df in list(comp_dfs.values())]
    latent_space = encoder.predict(data)

    return model, encoder, latent_space


class Modeller:
    def __init__(self, output_path, n_centroids=None):
        self.type = None
        self.output = output_path
        self.components = {}
        self.latent = {}
        self.score = {"Adjusted mutual information score": {}, "Silhouette Score": {}}
        if type == "VAE":
            self.pi = None
            self.mu_c = None
            self.var_c = None
        self.n_centroids = 3
        self.model = {}
        self.encoder = {}
        self.predicted_labels = {}
        self.top_features = {}
        self.tw_resolution = {}

    def add_component(self, name, comp, test_split=None):
        self.components[name] = comp

    def add_train_test(self, test_split):
        keys_copy = list(self.components.keys())
        for i, c in enumerate(keys_copy):
            if i == 0:
                (
                    self.components[f"{c}_train"],
                    self.components[f"{c}_test"],
                ) = train_test_split(
                    self.components[c],
                    test_size=test_split,
                    stratify=self.components[c]["Annotation"],
                )
                inds_train = self.components[f"{c}_train"].index
                inds_test = self.components[f"{c}_test"].index
            else:
                self.components[f"{c}_train"] = self.components[c].loc[inds_train]
                self.components[f"{c}_test"] = self.components[c].loc[inds_test]

    def fit(self, type, name, hp, verbose=1, build_only=False):
        if type != "AE" and type != "VAE":
            raise ValueError(f"{type} is not a valid model type.")
        model, encoder, latent_space = fit_comps(
            self,
            name,
            type,
            self.components,
            hp,
            self.output,
            self.n_centroids,
            verbose=verbose,
            build_only=build_only
        )
        self.model[f"{name}_{type}"] = model
        self.encoder[f"{name}_{type}"] = encoder
        self.latent[f"{name}_{type}"] = latent_space

    def plot_umap(
        self,
        type,
        name,
        n_neighbors,
        resolution,
        min_dist=0,
        verbose=1,
        tweak_resolution=False,
        k_classes=None,
        raw=False,
    ):

        model_umap = umap.UMAP(
            n_neighbors=n_neighbors, min_dist=min_dist, n_components=2
        )

        if raw:
            if "Multiomics_Integration" in name:
                embedding = model_umap.fit_transform(pd.concat(list(self.components.values()), axis=1).drop(columns="Annotation"))

            else:
                embedding = model_umap.fit_transform(self.components[f"{name}"].drop(columns="Annotation"))
        else:
            embedding = model_umap.fit_transform(self.latent[f"{name}_{type}"])

        sc.settings.verbosity = verbose
        new_clusters = anndata.AnnData(embedding)
        sc.pp.neighbors(new_clusters, n_neighbors=n_neighbors, use_rep="X")
        sc.tl.leiden(new_clusters, resolution=resolution)

        if tweak_resolution:
            tweak_amis = []
            tweak_resolution = []
            for k in range(k_classes[0], k_classes[1]+1):
                i_res = resolution
                while (
                    new_clusters.obs["leiden"].nunique() > k and i_res > 1e-10
                ):
                    i_res *= 0.9
                    sc.tl.leiden(new_clusters, resolution=i_res)

                tweak_resolution.append(i_res)

                df_plt = pd.DataFrame(embedding, columns=["UMAP 1", "UMAP 2"])
                df_plt["Leiden clusters"] = new_clusters.obs["leiden"].astype(int).to_list()
                if "Multiomics_Integration" in name:
                    df_plt["Cell type"] = self.components[list(self.components.keys())[0]][
                        "Annotation"
                    ].to_list()
                else:
                    df_plt["Cell type"] = self.components[name]["Annotation"].to_list()
                df_plt = df_plt.sort_values("Cell type")

                amis = metrics.adjusted_mutual_info_score(
                    labels_true=df_plt["Cell type"].to_list(),
                    labels_pred=df_plt["Leiden clusters"].to_list(),
                )
                tweak_amis.append(amis)
                ind_best = tweak_amis.index(max(tweak_amis))
                best_res = tweak_resolution[ind_best]
                sc.tl.leiden(new_clusters, resolution=best_res)

        df_plt = pd.DataFrame(embedding, columns=["UMAP 1", "UMAP 2"])
        df_plt["Leiden clusters"] = new_clusters.obs["leiden"].astype(int).to_list()
        if "Multiomics_Integration" in name:
            df_plt["Cell type"] = self.components[list(self.components.keys())[0]][
                "Annotation"
            ].to_list()
        else:
            df_plt["Cell type"] = self.components[name]["Annotation"].to_list()
        df_plt = df_plt.sort_values("Cell type")

        amis = metrics.adjusted_mutual_info_score(
            labels_true=df_plt["Cell type"].to_list(),
            labels_pred=df_plt["Leiden clusters"].to_list(),
        )

        ss = metrics.silhouette_score(embedding, df_plt["Cell type"])
        if verbose == 1:
            print(f"Cluster similarity: {amis:.4f}")
            print(f"Silhouette score: {ss:.4f}")

        self.score["Adjusted mutual information score"][f"{name}_{type}"] = amis
        self.score["Silhouette Score"][f"{name}_{type}"] = ss

        fig, ax = plt.subplots(ncols=2, tight_layout=True, figsize=(12, 4))

        sns.scatterplot(
            x="UMAP 1",
            y="UMAP 2",
            hue="Leiden clusters",
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
        ax[0].set_title(f"{type} {name} Leiden clustering")

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
        ax[1].set_title(f"{type} {name} reference annotation")
        plt.savefig(
            f"{self.output}{type}_{name}_combined_umap.png", dpi=300,
        )
        plt.show()
        plt.close()

    def transfer_label(self, name, type, k_classes, hp, gene_loc_df=None, explain=True, lr=0.01, epochs=150):
        if "Multiomics_Integration" in name:
            x_train = {
                k: v
                for k, v in zip(
                    [k for k in list(self.components.keys()) if "_train" in k],
                    [
                        self.components.get(k)
                        for k in self.components.keys()
                        if "_train" in k
                    ],
                )
            }
            x_test = {
                k: v
                for k, v in zip(
                    [k for k in list(self.components.keys()) if "_test" in k],
                    [
                        self.components.get(k)
                        for k in self.components.keys()
                        if "_test" in k
                    ],
                )
            }
            y_train = pd.factorize(
                [
                    self.components.get(k)
                    for k in self.components.keys()
                    if "_train" in k
                ][0]["Annotation"],
                sort=True,
            )
            test_mat = [v for c, v in list(self.components.items()) if "_test" in c]
        else:
            x_train = {f"{name}_train": self.components[f"{name}_train"]}
            x_test = {f"{name}_test": self.components[f"{name}_test"]}
            y_train = pd.factorize(
                self.components[f"{name}_train"]["Annotation"], sort=True
            )
            test_mat = [self.components[f"{name}_test"]]

        y_test = pd.factorize(test_mat[0]["Annotation"], sort=True)

        hp["epochs"] = 1
        self.fit(type=type, name=f"{name}_train", hp=hp, verbose=0, build_only=True)
        enc = self.encoder[f"{name}_train_{type}"]
        enc.layers.remove(enc.get_layer("Dropout_latent"))
        class_layer = Dense(k_classes, name="Classes", activation="sigmoid")(enc.layers[-1].output)
        train_clf = Model(inputs=enc.inputs, outputs=[class_layer])

        class_weights = pd.Series(y_train[0]).value_counts().to_dict()
        class_weights = {k:1/(v/len(y_train[0])) for k,v in class_weights.items()}
        sample_weights = [class_weights[x] for x in y_train[0]]
        sample_weights = np.array([x/sum(sample_weights) for x in sample_weights])

        train_clf.compile(
            optimizer=Adam(learning_rate=lr),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
        )
        train_clf.fit(
            list(v.iloc[:, :-1].values for v in x_train.values()), y_train[0], epochs=epochs,
            sample_weight=sample_weights,
            validation_data=(list(v.iloc[:, :-1].values for v in x_test.values()), y_test[0]),
            callbacks=[EarlyStopping(monitor="val_sparse_categorical_accuracy", patience=50, restore_best_weights=True, min_delta=0.0002)]
        )
        pred = np.argmax(
            train_clf.predict(list(v.iloc[:, :-1].values for v in x_test.values())),
            axis=1,
        )

        if explain:
            if "Multiomics_Integration" in name:
                exp_train = x_train[f"cDNA_train"].iloc[:, :-1].join(x_train[f"Chromatin_train"].iloc[:, :-1]).values
                exp_feature_names = x_train[f"cDNA_train"].iloc[:, :-1].columns.to_list() + x_train[f"Chromatin_train"].iloc[:, :-1].columns.to_list()
                exp_class_names = np.unique(
                    x_train[f"cDNA_train"].iloc[:, -1].values
                ).tolist()
            else:
                exp_train = x_train[f"{name}_train"].iloc[:, :-1].values
                exp_feature_names = x_train[f"{name}_train"].iloc[:, :-1].columns.to_list()
                exp_class_names = np.unique(
                    x_train[f"{name}_train"].iloc[:, -1].values
                ).tolist()

            explainer = lime.lime_tabular.LimeTabularExplainer(
                exp_train,
                feature_names=exp_feature_names,
                class_names=exp_class_names,
                discretize_continuous=True,
            )
            pred_labels = y_test[1][pred]
            top = []
            if "Multiomics_Integration" in name:
                lab_groups = x_test["cDNA_test"][
                    x_test["cDNA_test"]["Annotation"] == pred_labels
                ].groupby("Annotation")
            else:
                lab_groups = x_test[f"{name}_test"][
                    x_test[f"{name}_test"]["Annotation"] == pred_labels
                    ].groupby("Annotation")
            labs = []
            for lab, grp in lab_groups:
                labs.append(lab)
                print("Currently at: ", lab)
                ind = grp.sample(n=1)["Annotation"].index
                if "Multiomics_Integration" in name:
                    def predict_splitter(concat_frame):
                        if not concat_frame.shape[0] == 5000:
                            data = np.expand_dims(concat_frame, axis=0)
                            normal_pred = train_clf.predict(np.array_split(data, [len(x_test["cDNA_test"].columns)-1], axis=1))
                            normal_pred = np.squeeze(normal_pred.reshape(-1,1))
                        else:
                            normal_pred = train_clf.predict(np.array_split(concat_frame, [len(x_test["cDNA_test"].columns) - 1], axis=1))
                        return normal_pred

                    exp = explainer.explain_instance(
                        np.squeeze(x_test[f"cDNA_test"].iloc[:, :-1].join(x_test[f"Chromatin_test"].iloc[:, :-1]).loc[ind].values),
                        # np.squeeze(x_test[f"{name}_test"].loc[ind].iloc[:, :-1].values),
                        predict_splitter,
                        num_features=3,
                        top_labels=k_classes,
                    )
                else:
                    exp = explainer.explain_instance(
                        np.squeeze(x_test[f"{name}_test"].loc[ind].iloc[:, :-1].values),
                        train_clf.predict,
                        num_features=3,
                        top_labels=k_classes,
                    )

                markers = exp.as_list(label=y_test[1].get_loc(lab))
                for i in markers:
                    if not "Multiomics_Integration" in name:
                        top.append(*[x for x in i[0].split(" ") if re.search('[a-zA-Z]', x)])
                    else :
                        features = [x for x in i[0].split(" ") if re.search('[a-zA-Z]', x)]
                        for f in features:
                            if f in self.components["cDNA"].columns:
                                top.append(f)
                            else:
                                top.append(f)

            if "Multiomics_Integration" in name:
                plot_data = self.components[f"cDNA"].iloc[:, :-1].join(self.components[f"Chromatin"]).sort_values("Annotation")
            else:
                plot_data = self.components[f"{name}"].sort_values("Annotation")
            fig, ax = plt.subplots(1,1, figsize=(16,8))
            fig.subplots_adjust(left=.5)

            maxlen_top = len(max(top, key=len))
            padded_top = [f"{x.rjust(maxlen_top)}" for x in top]
            heatmap_df = plot_data[top + ["Annotation"]].groupby("Annotation").mean()
            heatmap_df.rename(columns={k:v for k,v in zip(top, padded_top)}, inplace=True)
            sns.heatmap(heatmap_df.T, linewidths=0.25, cmap="GnBu")

            groups = {k:tuple(padded_top[i*3:(i*3)+3]) for i, k in enumerate(labs)}
            heatmap_helper.annotate_yranges(groups=groups)

            plt.savefig(
                f"{self.output}{type}_{name}_top_features_matrix_plot.png", dpi=300, bbox_inches="tight"
            )
            plt.show()
            plt.close()

        cm = pd.crosstab(y_test[0], pred)
        cm.index = y_test[1].tolist()
        cm.columns = y_test[1][np.unique(pred)]
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

        for i in range(0, k_classes):
            ax.add_patch(Rectangle((i, i), 1, 1, fill=False, edgecolor="black", lw=2))

        plt.xlabel("Predicted labels", fontsize=16)
        plt.ylabel("Reference labels", fontsize=16)
        plt.title(f"Contingency matrix {type} {name}_test", fontsize=20)
        plt.savefig(
            f"{self.output}{type}_{name}_test_contingency_matrix.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.show()
        plt.close()

        precision = metrics.classification_report(
            y_true=y_test[0], y_pred=pred, zero_division=0, output_dict=True
        )["weighted avg"]["f1-score"]
        print(precision)
        if "Label transfer accuracy" not in self.score:
            self.score["Label transfer accuracy"] = {}
        self.score["Label transfer accuracy"][f"{name}_{type}"] = np.round(precision, 4)

    def evaluate(self):
        pd.set_option("display.max_rows", None)
        pd.set_option("display.max_columns", None)
        pd.set_option("display.width", None)
        print(pd.DataFrame(self.score))
        pd.DataFrame(self.score).to_csv(f"{self.output}evaluation_{time.strftime('%Y%m%d')}.csv")

    def auto_tweak(self, name, type, tweak_hp, n_neighbors, resolution, k_classes):
        multi = "Multiomics_Integration" in name
        geom_dims_comps = {
            name: np.ceil(
                np.geomspace(
                    tweak_hp["min_max_dim_comps"][name][0],
                    tweak_hp["min_max_dim_comps"][name][1],
                    tweak_hp["max_layers"],
                )
            )
            .astype(int)
            .tolist()
            for name in tweak_hp["min_max_dim_comps"]
        }
        combs = {}
        for c in tweak_hp["min_max_dim_comps"]:
            combs[c] = []
            for layers in range(tweak_hp["min_layers"], tweak_hp["max_layers"] + 1):
                combs[c].append(
                    list(itertools.combinations(geom_dims_comps[c], layers))
                )
            combs[c] = [val for sublist in combs[c] for val in sublist]
        paras = list(itertools.product(*combs.values()))

        cols = [f"{x}_layers" for x in tweak_hp["min_max_dim_comps"].keys()] + [
            "AMIS",
            "SS",
            "Optimal Resolution",
        ]
        output_df = pd.DataFrame(columns=cols)
        bar = progressbar.ProgressBar()
        for setting in bar(paras):
            if multi:
                hp = {
                    "act_encoder": "relu",
                    "act_decoder": "relu",
                    "act_restore": "sigmoid",
                    "epochs": tweak_hp["epochs"],
                    "layer_dims": {
                        n: p
                        for n, p in zip(
                            tweak_hp["min_max_dim_comps"],
                            [list(l) for l in setting[:-1]],
                        )
                    },
                    "layer_dims_post_merge": list(setting[-1]),
                    "dropout_input": tweak_hp["dropout_input"],
                    "dropout_latent": tweak_hp["dropout_latent"],
                    "learning_rate": tweak_hp["learning_rate"],
                    "loss": tweak_hp["loss"],
                }
            else:
                hp = {
                    "act_encoder": "relu",
                    "act_decoder": "relu",
                    "act_restore": "sigmoid",
                    "epochs": tweak_hp["epochs"],
                    "layer_dims": {
                        n: p
                        for n, p in zip(
                            tweak_hp["min_max_dim_comps"], [list(l) for l in setting]
                        )
                    },
                    "dropout_input": tweak_hp["dropout_input"],
                    "dropout_latent": tweak_hp["dropout_latent"],
                    "learning_rate": tweak_hp["learning_rate"],
                    "loss": tweak_hp["loss"],
                }
            try:
                self.fit(type, name, hp, verbose=0)
                self.plot_umap(
                    type,
                    name,
                    n_neighbors,
                    resolution,
                    verbose=0,
                    tweak_resolution=True,
                    k_classes=k_classes,
                )
            except TypeError:
                continue
            amis = self.score["Adjusted mutual information score"][f"{name}_{type}"]
            ss = self.score["Silhouette Score"][f"{name}_{type}"]
            tw_resolution = self.tw_resolution[f"{name}_{type}"]
            output_df.loc[len(output_df)] = [l for l in setting] + [
                amis,
                ss,
                tw_resolution,
            ]
            output_df.sort_values("SS").to_csv(
                f"{self.output}{type}_{name}_tweaking.csv"
            )
