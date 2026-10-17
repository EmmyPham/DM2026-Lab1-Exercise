"""Tools for exploring relationships and similarities in text datasets."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from langchain_core.tools import tool


def make_tools(session):
    @tool
    def cosine_similarity_tool(doc_id_1: int, doc_id_2: int) -> dict:
        """Compute cosine similarity between two documents using raw term count vectors in the DTM.

        Args:
            doc_id_1: Row index of the first document in the dataset.
            doc_id_2: Row index of the second document in the dataset.

        Returns:
            dict: Summary containing result_id, doc indices, previews, and cosine similarity score.
        """
        if session.feature_matrix is None:
            return {
                "error": "No feature matrix found in session. Run build_dtm_tool first."
            }

        matrix = session.feature_matrix
        n_docs = matrix.shape[0]

        if doc_id_1 < 0 or doc_id_1 >= n_docs or doc_id_2 < 0 or doc_id_2 >= n_docs:
            return {
                "error": f"Document index out of bounds. Dataset has {n_docs} documents (indices 0 to {n_docs - 1})."
            }

        # Extract raw count vectors for doc_id_1 and doc_id_2
        if hasattr(matrix, "toarray"):
            v1 = matrix[doc_id_1].toarray().ravel()
            v2 = matrix[doc_id_2].toarray().ravel()
        else:
            v1 = np.ravel(matrix[doc_id_1])
            v2 = np.ravel(matrix[doc_id_2])

        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)

        # Handle zero-vector cases (empty text rows) safely
        if norm1 == 0.0 or norm2 == 0.0:
            similarity = 0.0
        else:
            similarity = float(np.dot(v1, v2) / (norm1 * norm2))

        # Get document text previews from session.dataframe if available
        doc_1_preview = ""
        doc_2_preview = ""
        if session.dataframe is not None and "text" in session.dataframe.columns:
            doc_1_preview = str(session.dataframe["text"].iloc[doc_id_1])[:100]
            doc_2_preview = str(session.dataframe["text"].iloc[doc_id_2])[:100]

        result_id = session.next_result_id("cosine")
        summary = {
            "result_id": result_id,
            "doc_id_1": doc_id_1,
            "doc_id_2": doc_id_2,
            "similarity": round(similarity, 4),
            "doc_1_preview": doc_1_preview,
            "doc_2_preview": doc_2_preview,
        }

        session.store_result("cosine_similarity_tool", {"doc_id_1": doc_id_1, "doc_id_2": doc_id_2}, summary)
        return summary

    @tool
    def feature_correlation_matrix_tool(top_n: int = 20) -> dict:
        """Compute feature-vs-feature Pearson correlation matrix for top-N terms by variance and render a heatmap.

        Args:
            top_n: Number of terms with highest variance to include (default 20).

        Returns:
            dict: Summary containing result_id, top_n, list of term names, and correlation matrix.
        """
        if top_n <= 0:
            return {"error": "top_n must be a positive integer greater than 0."}

        if session.feature_matrix is None or session.feature_names is None or session.feature_matrix.shape[0] == 0 or session.feature_matrix.shape[1] == 0:
            return {
                "error": "Feature matrix is empty or not found in session. Run build_dtm_tool first."
            }

        matrix = session.feature_matrix
        feature_names = np.array(session.feature_names)

        # Compute term variances directly on sparse matrix
        if hasattr(matrix, "toarray"):
            mean_x = np.ravel(matrix.mean(axis=0))
            mean_sq_x = np.ravel(matrix.power(2).mean(axis=0))
            variances = mean_sq_x - mean_x ** 2
        else:
            variances = np.var(matrix, axis=0)

        n_features = len(feature_names)
        actual_top_n = min(top_n, n_features)

        # Select top-N indices by variance (descending)
        top_indices = np.argsort(variances)[::-1][:actual_top_n]
        top_terms = feature_names[top_indices].tolist()

        # Densify ONLY the selected top-N columns, avoiding converting full matrix to dense
        if hasattr(matrix, "toarray"):
            top_matrix_dense = matrix[:, top_indices].toarray()
        else:
            top_matrix_dense = matrix[:, top_indices]

        # Compute Pearson correlation matrix across top terms (columns)
        corr_matrix = np.corrcoef(top_matrix_dense.T)

        # Handle single feature edge case where corrcoef returns scalar or 1x1 array
        if corr_matrix.ndim == 0:
            corr_matrix = np.array([[1.0]])

        # Replace NaN with 0.0 if any constant terms present
        corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)

        # Render seaborn heatmap figure
        fig, ax = plt.subplots(figsize=(max(6, actual_top_n * 0.8), max(5, actual_top_n * 0.7)))
        sns.heatmap(
            corr_matrix,
            annot=True,
            fmt=".2f",
            cmap="coolwarm",
            xticklabels=top_terms,
            yticklabels=top_terms,
            ax=ax,
            vmin=-1,
            vmax=1
        )
        ax.set_title(f"Feature Correlation Matrix (Top {actual_top_n} by Variance)")
        plt.tight_layout()

        session.pending_figure = fig
        plt.close(fig)

        # Round matrix for JSON summary output
        corr_list = np.round(corr_matrix, 4).tolist()

        result_id = session.next_result_id("feature_corr")
        summary = {
            "result_id": result_id,
            "top_n": top_n,
            "evaluated_terms_count": actual_top_n,
            "terms": top_terms,
            "correlation_matrix": corr_list,
        }

        session.store_result("feature_correlation_matrix_tool", {"top_n": top_n}, summary)
        return summary

    return [cosine_similarity_tool, feature_correlation_matrix_tool]
