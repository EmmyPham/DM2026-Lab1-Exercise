"""Tools for dimensionality reduction and label binarization."""

from typing import Literal, Optional
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap

from langchain_core.tools import tool


def make_tools(session):
    @tool
    def reduce_dimensions_tool(
        method: Literal["pca", "tsne", "umap"] = "pca",
        n_components: int = 2,
        perplexity: float = 30.0,
        n_neighbors: int = 15,
        random_state: Optional[int] = None
    ) -> dict:
        """Reduce dimensionality of the Document-Term Matrix (PCA, t-SNE, or UMAP) and plot the 2D coordinates.

        Args:
            method: Reduction algorithm to use ('pca', 'tsne', or 'umap'). Default is 'pca'.
            n_components: Target dimension (default 2).
            perplexity: Perplexity parameter for t-SNE (default 30.0). Adjust for small datasets.
            n_neighbors: Number of neighbors parameter for UMAP (default 15). Adjust for small datasets.
            random_state: Seed for random number generator to ensure reproducible results.

        Returns:
            dict: Summary containing result_id, method, coordinates preview, and explained variance ratio (PCA).
        """
        if session.feature_matrix is None:
            return {
                "error": "No feature matrix found in session. Run build_dtm_tool first."
            }

        matrix = session.feature_matrix
        if hasattr(matrix, "toarray"):
            X = matrix.toarray()
        else:
            X = matrix

        n_samples = X.shape[0]
        if n_samples == 0:
            return {"error": "Feature matrix is empty."}

        explained_variance_ratio = None
        note = None

        if method == "pca":
            reducer = PCA(n_components=n_components, random_state=random_state)
            coords = reducer.fit_transform(X)
            explained_variance_ratio = np.round(reducer.explained_variance_ratio_, 4).tolist()

        elif method == "tsne":
            if perplexity >= n_samples:
                actual_perplexity = max(1.0, float(n_samples - 1))
                note = f"Warning: perplexity={perplexity} was >= n_samples ({n_samples}). Auto-adjusted perplexity to {actual_perplexity}."
            else:
                actual_perplexity = perplexity

            reducer = TSNE(
                n_components=n_components,
                perplexity=actual_perplexity,
                init="pca",
                random_state=random_state
            )
            coords = reducer.fit_transform(X)

        elif method == "umap":
            if n_neighbors >= n_samples:
                actual_n_neighbors = max(2, n_samples - 1)
                note = f"Warning: n_neighbors={n_neighbors} was >= n_samples ({n_samples}). Auto-adjusted n_neighbors to {actual_n_neighbors}."
            else:
                actual_n_neighbors = n_neighbors

            reducer = umap.UMAP(
                n_components=n_components,
                n_neighbors=actual_n_neighbors,
                random_state=random_state
            )
            coords = reducer.fit_transform(X)

        else:
            return {"error": f"Unknown dimensionality reduction method '{method}'."}

        # Store full unrounded coordinates in session.artifacts
        artifact_key = f"reduced_{method}"
        session.artifacts[artifact_key] = coords

        # Obtain category labels for coloring the scatter plot
        category_labels = None
        if session.dataframe is not None:
            if "category_name" in session.dataframe.columns:
                category_labels = session.dataframe["category_name"].values
            elif "category" in session.dataframe.columns:
                category_labels = session.dataframe["category"].values

        if category_labels is None and session.labels is not None:
            category_labels = session.labels

        if category_labels is None:
            category_labels = np.array(["Document"] * n_samples)

        # Plot 2D scatter plot colored by category
        fig, ax = plt.subplots(figsize=(8, 6))
        plot_df = pd.DataFrame({
            "Dim 1": coords[:, 0],
            "Dim 2": coords[:, 1] if n_components > 1 else np.zeros(n_samples),
            "Category": category_labels
        })

        sns.scatterplot(
            data=plot_df,
            x="Dim 1",
            y="Dim 2",
            hue="Category",
            style="Category",
            s=100,
            ax=ax
        )
        ax.set_title(f"Dimensionality Reduction: {method.upper()}")
        plt.tight_layout()

        # Mandatory plotting requirement
        session.pending_figure = fig
        plt.close(fig)

        coords_rounded = np.round(coords, 4).tolist()

        result_id = session.next_result_id("reduce_dim")
        summary = {
            "result_id": result_id,
            "method": method,
            "n_components": n_components,
            "total_samples": n_samples,
            "explained_variance_ratio": explained_variance_ratio,
            "coordinates": coords_rounded[:20],
        }
        if note:
            summary["note"] = note

        session.store_result("reduce_dimensions_tool", {"method": method, "n_components": n_components}, summary)
        return summary

    @tool
    def binarize_labels_tool() -> dict:
        """Binarize category labels into a one-hot encoded matrix.

        Returns:
            dict: Summary containing result_id, n_samples, n_classes, classes list, and binarized labels preview.
        """
        if session.dataframe is None or session.dataframe.empty:
            return {"error": "No DataFrame found in session. Load data first."}

        # Extract target category labels, handling missing values
        if "category_name" in session.dataframe.columns:
            labels_s = session.dataframe["category_name"].fillna("unknown").astype(str)
        elif "category" in session.dataframe.columns:
            labels_s = session.dataframe["category"].fillna(-1).astype(str)
        elif session.labels is not None:
            labels_s = pd.Series(session.labels).fillna("unknown").astype(str)
        else:
            return {"error": "No category labels found in session."}

        labels = labels_s.values
        unique_classes = np.unique(labels)
        n_classes = len(unique_classes)

        if n_classes == 0:
            return {"error": "No unique categories found to binarize."}

        # One-hot encoding matrix (1 column per category)
        one_hot_df = pd.get_dummies(labels_s)[unique_classes]
        binarized_matrix = one_hot_df.values.astype(int)

        # Store in session.artifacts
        session.artifacts["binarized_labels"] = binarized_matrix
        session.artifacts["binarizer_classes"] = unique_classes.tolist()

        result_id = session.next_result_id("binarize")
        summary = {
            "result_id": result_id,
            "n_samples": len(labels),
            "n_classes": n_classes,
            "classes": unique_classes.tolist(),
            "binarized_labels_preview": binarized_matrix[:10].tolist()
        }

        session.store_result("binarize_labels_tool", {}, summary)
        return summary

    return [reduce_dimensions_tool, binarize_labels_tool]
