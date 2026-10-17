"""Tools for filtering features in the Document-Term Matrix (DTM)."""

from typing import Optional
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from langchain_core.tools import tool


def make_tools(session):
    @tool
    def variance_filter_tool(threshold: float = 0.0) -> dict:
        """Filter terms from the Document-Term Matrix based on feature variance.

        Terms with variance strictly greater than the threshold are kept; terms
        with variance less than or equal to the threshold are flagged for removal.

        Args:
            threshold: Minimum variance threshold (default 0.0). Terms must have
                variance strictly greater than this value to be kept.

        Returns:
            dict: Summary containing result_id, threshold, feature counts,
            and previews of kept and removed feature names (sorted by variance descending).
        """
        if session.feature_matrix is None or session.feature_names is None:
            return {
                "error": "No feature matrix found in session. Run build_dtm_tool first."
            }

        matrix = session.feature_matrix

        # Compute variance per feature across all documents directly on sparse matrix or numpy array
        if hasattr(matrix, "toarray"):
            # E(X^2) - (E(X))^2 for sparse matrix without dense array conversion
            mean = np.ravel(matrix.mean(axis=0))
            mean_sq = np.ravel(matrix.power(2).mean(axis=0))
            variances = mean_sq - mean ** 2
        else:
            variances = np.var(matrix, axis=0)

        feature_names = np.array(session.feature_names)

        # Build full report DataFrame for visualization (term as column 1, variance as column 2)
        full_report = pd.DataFrame({
            "term": feature_names,
            "variance": variances
        }).sort_values(by="variance", ascending=False).reset_index(drop=True)

        kept_df = full_report[full_report["variance"] > threshold]
        removed_df = full_report[full_report["variance"] <= threshold]

        kept_features_sorted = kept_df["term"].tolist()
        removed_features_sorted = removed_df["term"].tolist()

        result_id = session.next_result_id("variance")
        summary = {
            "result_id": result_id,
            "threshold": threshold,
            "total_features": len(feature_names),
            "kept_features_count": len(kept_features_sorted),
            "removed_features_count": len(removed_features_sorted),
            "kept_features": kept_features_sorted[:20],
            "removed_features": removed_features_sorted[:20],
        }

        session.store_result("variance_filter_tool", {"threshold": threshold}, summary, full_report=full_report)
        return summary

    @tool
    def pearson_filter_tool(target_class: str, threshold: float = 0.0) -> dict:
        """Filter terms from the Document-Term Matrix based on Pearson correlation with a target class.

        Terms with absolute Pearson correlation |r| strictly greater than threshold are kept.

        Args:
            target_class: Target class category name (e.g. 'catB') for one-vs-rest binary correlation.
            threshold: Minimum absolute correlation threshold (default 0.0). Terms with
                |pearson_r| strictly greater than this value are kept.

        Returns:
            dict: Summary containing result_id, target_class, threshold, feature counts,
            and previews of kept and removed feature names.
        """
        if session.feature_matrix is None or session.feature_names is None:
            return {
                "error": "No feature matrix found in session. Run build_dtm_tool first."
            }

        # Obtain labels per document
        if session.dataframe is not None and "category_name" in session.dataframe.columns:
            labels = session.dataframe["category_name"].values
        elif session.labels is not None:
            labels = session.labels
        else:
            return {"error": "No document category labels found in session."}

        # Binary indicator vector for target_class
        y = (labels == target_class).astype(float)
        std_y = np.std(y)
        if std_y == 0:
            return {"error": f"Target class '{target_class}' does not exist or has no variation in labels."}

        matrix = session.feature_matrix
        feature_names = np.array(session.feature_names)
        n_samples = matrix.shape[0]

        # Fully vectorized computation across the whole vocabulary using matrix operations
        y_centered = y - np.mean(y)

        if hasattr(matrix, "toarray"):
            # Vectorized E(X) and std(X)
            mean_x = np.ravel(matrix.mean(axis=0))
            mean_sq_x = np.ravel(matrix.power(2).mean(axis=0))
            var_x = mean_sq_x - mean_x ** 2
            std_x = np.sqrt(np.maximum(var_x, 0.0))

            # Covariance: (X^T @ y_centered) / n_samples
            cov = np.ravel(matrix.T.dot(y_centered)) / n_samples
        else:
            mean_x = np.mean(matrix, axis=0)
            std_x = np.std(matrix, axis=0)
            cov = np.mean((matrix - mean_x) * y_centered[:, None], axis=0)

        # Handle zero std_x safely
        with np.errstate(divide="ignore", invalid="ignore"):
            correlations = np.where(std_x > 0, cov / (std_x * std_y), 0.0)

        correlations = np.nan_to_num(correlations, nan=0.0)

        # Build full report DataFrame, sorted by absolute correlation |r| descending
        full_report = pd.DataFrame({
            "term": feature_names,
            "pearson_r": correlations,
            "abs_r": np.abs(correlations)
        }).sort_values(by="abs_r", ascending=False).reset_index(drop=True)

        kept_df = full_report[full_report["abs_r"] > threshold]
        removed_df = full_report[full_report["abs_r"] <= threshold]

        kept_features_sorted = kept_df["term"].tolist()
        removed_features_sorted = removed_df["term"].tolist()

        # Drop helper column abs_r from exported report, preserving raw pearson_r values
        full_report_export = full_report[["term", "pearson_r"]]

        result_id = session.next_result_id("pearson")
        summary = {
            "result_id": result_id,
            "target_class": target_class,
            "threshold": threshold,
            "total_features": len(feature_names),
            "kept_features_count": len(kept_features_sorted),
            "removed_features_count": len(removed_features_sorted),
            "kept_features": kept_features_sorted[:20],
            "removed_features": removed_features_sorted[:20],
        }

        session.store_result("pearson_filter_tool", {"target_class": target_class, "threshold": threshold}, summary, full_report=full_report_export)
        return summary

    @tool
    def spearman_filter_tool(
        target_class: str,
        threshold: float = 0.0,
        max_terms: Optional[int] = None
    ) -> dict:
        """Filter terms from the Document-Term Matrix based on Spearman rank correlation with a target class.

        Terms with absolute Spearman correlation |spearman_r| strictly greater than threshold are kept.

        Args:
            target_class: Target class category name (e.g. 'catB') for one-vs-rest binary correlation.
            threshold: Minimum absolute correlation threshold (default 0.0). Terms with
                |spearman_r| strictly greater than this value are kept.
            max_terms: Optional maximum number of terms with the highest variance
                to compute Spearman correlation for. If None, computes for all terms.

        Returns:
            dict: Summary containing result_id, target_class, threshold, feature counts,
            and previews of kept and removed feature names.
        """
        if session.feature_matrix is None or session.feature_names is None:
            return {
                "error": "No feature matrix found in session. Run build_dtm_tool first."
            }

        # Obtain labels per document
        if session.dataframe is not None and "category_name" in session.dataframe.columns:
            labels = session.dataframe["category_name"].values
        elif session.labels is not None:
            labels = session.labels
        else:
            return {"error": "No document category labels found in session."}

        # Binary indicator vector for target_class
        y = (labels == target_class).astype(float)
        if np.std(y) == 0:
            return {"error": f"Target class '{target_class}' does not exist or has no variation in labels."}

        matrix = session.feature_matrix
        feature_names = np.array(session.feature_names)
        dtm_total_features = len(feature_names)

        # Compute feature variances to select top terms by variance if max_terms is specified
        if hasattr(matrix, "toarray"):
            mean_x = np.ravel(matrix.mean(axis=0))
            mean_sq_x = np.ravel(matrix.power(2).mean(axis=0))
            variances = mean_sq_x - mean_x ** 2
        else:
            variances = np.var(matrix, axis=0)

        if max_terms is not None and max_terms < dtm_total_features:
            # Select top max_terms indices by variance
            selected_indices = np.argsort(variances)[::-1][:max_terms]
        else:
            selected_indices = np.arange(dtm_total_features)

        selected_features = feature_names[selected_indices]

        # Extract selected columns
        if hasattr(matrix, "toarray"):
            sub_matrix = matrix[:, selected_indices].toarray()
        else:
            sub_matrix = matrix[:, selected_indices]

        # Rank transform y
        y_ranked = rankdata(y)
        y_ranked_centered = y_ranked - np.mean(y_ranked)
        std_y_ranked = np.std(y_ranked)

        # Compute rank matrix for selected features across documents
        ranks_x = np.apply_along_axis(rankdata, axis=0, arr=sub_matrix)

        mean_x_ranked = np.mean(ranks_x, axis=0)
        std_x_ranked = np.std(ranks_x, axis=0)

        # Covariance between ranked X and ranked y
        cov = np.mean((ranks_x - mean_x_ranked) * y_ranked_centered[:, None], axis=0)

        with np.errstate(divide="ignore", invalid="ignore"):
            correlations = np.where(std_x_ranked > 0, cov / (std_x_ranked * std_y_ranked), 0.0)

        correlations = np.nan_to_num(correlations, nan=0.0)

        # Build full report DataFrame for evaluated terms, sorted by absolute correlation |spearman_r| descending
        full_report = pd.DataFrame({
            "term": selected_features,
            "spearman_r": correlations,
            "abs_r": np.abs(correlations)
        }).sort_values(by="abs_r", ascending=False).reset_index(drop=True)

        kept_df = full_report[full_report["abs_r"] > threshold]
        removed_df = full_report[full_report["abs_r"] <= threshold]

        kept_features_sorted = kept_df["term"].tolist()
        removed_features_sorted = removed_df["term"].tolist()

        # Drop helper column abs_r from exported report, preserving raw spearman_r values
        full_report_export = full_report[["term", "spearman_r"]]

        result_id = session.next_result_id("spearman")
        summary = {
            "result_id": result_id,
            "target_class": target_class,
            "threshold": threshold,
            "dtm_total_features": dtm_total_features,
            "evaluated_features_count": len(selected_features),
            "max_terms_filter": max_terms,
            "kept_features_count": len(kept_features_sorted),
            "removed_features_count": len(removed_features_sorted),
            "kept_features": kept_features_sorted[:20],
            "removed_features": removed_features_sorted[:20],
            "note": f"Evaluated top {len(selected_features)} features by variance out of {dtm_total_features} total features in DTM." if max_terms is not None and max_terms < dtm_total_features else "Evaluated all features in DTM."
        }

        session.store_result("spearman_filter_tool", {"target_class": target_class, "threshold": threshold, "max_terms": max_terms}, summary, full_report=full_report_export)
        return summary

    return [variance_filter_tool, pearson_filter_tool, spearman_filter_tool]
