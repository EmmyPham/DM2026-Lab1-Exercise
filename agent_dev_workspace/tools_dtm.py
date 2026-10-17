from typing import List, Optional, Union
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer
from langchain_core.tools import tool


def make_tools(session):
    @tool
    def build_dtm_tool(
        ngram_range: List[int] = [1, 1],
        stop_words: Optional[Union[str, List[str]]] = None
    ) -> dict:
        """
        Builds a Document-Term Matrix (DTM) from session.dataframe['text'] using CountVectorizer.

        Stores the sparse matrix in session.feature_matrix, vocabulary in session.feature_names,
        and the fitted vectorizer in session.artifacts['count_vectorizer'].

        Args:
            ngram_range (List[int]): Lower and upper boundary of n-grams (e.g. [1, 1]). Default is [1, 1].
            stop_words (Optional[Union[str, List[str]]]): Stop words setting. Can be 'english' for scikit-learn's
                built-in list, a list of custom stop word strings, or None to keep all words.

        Returns:
            dict: Summary containing result_id, status, n_documents, n_terms, vocabulary (preview up to 10 terms),
                  non_zero, total_elements, and sparsity_pct.
        """
        if session.dataframe is None or session.dataframe.empty:
            return {
                "status": "error",
                "message": "No dataframe found in session. Please load a dataset first."
            }

        if "text" not in session.dataframe.columns:
            return {
                "status": "error",
                "message": "DataFrame must contain a 'text' column to build DTM."
            }

        # Convert list to tuple for CountVectorizer
        ngram_tuple = tuple(ngram_range) if ngram_range else (1, 1)

        vectorizer = CountVectorizer(
            ngram_range=ngram_tuple,
            stop_words=stop_words
        )

        texts = session.dataframe["text"].fillna("").astype(str)
        X = vectorizer.fit_transform(texts)
        feature_names = vectorizer.get_feature_names_out().tolist()

        # Update session state
        session.feature_matrix = X
        session.feature_names = feature_names
        session.artifacts["count_vectorizer"] = vectorizer

        # Synchronize labels and categories if category_name is present
        if "category_name" in session.dataframe.columns:
            session.set_labels(session.dataframe["category_name"].tolist())

        n_rows, n_cols = X.shape
        non_zero = int(X.nnz)
        total_elements = n_rows * n_cols
        sparsity_pct = float(round(100.0 * (1.0 - non_zero / total_elements), 4)) if total_elements > 0 else 0.0

        vocab_preview = feature_names[:10] if len(feature_names) > 10 else feature_names

        result_id = session.next_result_id("build_dtm")
        summary = {
            "result_id": result_id,
            "status": "success",
            "n_documents": n_rows,
            "n_terms": n_cols,
            "vocabulary": vocab_preview,
            "non_zero": non_zero,
            "total_elements": total_elements,
            "sparsity_pct": sparsity_pct
        }

        session.store_result(
            tool_name="build_dtm_tool",
            args={
                "ngram_range": ngram_range,
                "stop_words": stop_words
            },
            summary=summary
        )

        return summary

    @tool
    def term_frequency_tool(top_n: int = 10) -> dict:
        """
        Aggregates total term frequencies across all documents in session.feature_matrix,
        sorting terms by frequency descending.

        Args:
            top_n (int): Number of top frequent terms to return in the summary dictionary. Default is 10.

        Returns:
            dict: Summary containing result_id, status, total_terms, and top_terms_preview.
        """
        if session.feature_matrix is None or session.feature_names is None:
            return {
                "status": "error",
                "message": "No document-term matrix found in session. Please run build_dtm_tool first."
            }

        counts = np.asarray(session.feature_matrix.sum(axis=0)).flatten()
        term_freqs = {term: int(count) for term, count in zip(session.feature_names, counts)}

        # Sort terms by frequency descending
        sorted_term_freqs = dict(sorted(term_freqs.items(), key=lambda item: item[1], reverse=True))

        # Preview top_n terms for concise output
        top_terms_preview = dict(list(sorted_term_freqs.items())[:top_n])

        result_id = session.next_result_id("term_frequency")
        summary = {
            "result_id": result_id,
            "status": "success",
            "total_terms": len(session.feature_names),
            "top_terms_preview": top_terms_preview
        }

        # Store full report in session for visualization/downstream tools
        full_report_df = pd.DataFrame(
            list(sorted_term_freqs.items()),
            columns=["term", "frequency"]
        )

        session.store_result(
            tool_name="term_frequency_tool",
            args={"top_n": top_n},
            summary=summary,
            full_report=full_report_df
        )

        return summary

    @tool
    def dtm_heatmap_tool(n_terms: int = 20, n_documents: int = 20) -> dict:
        """
        Renders a heatmap of a raw positional slice of the global Document-Term Matrix (DTM).

        Slices session.feature_matrix[0:n_documents, 0:n_terms] and sets the resulting plot on
        session.pending_figure.

        Args:
            n_terms (int): Number of terms (columns) to include from vocabulary order. Default is 20.
            n_documents (int): Number of documents (rows) to include from document order. Default is 20.

        Returns:
            dict: Summary containing result_id, status, n_documents, n_terms, terms, and matrix_slice.
        """
        if session.feature_matrix is None or session.feature_names is None:
            return {
                "status": "error",
                "message": "No document-term matrix found in session. Please run build_dtm_tool first."
            }

        if n_terms <= 0 or n_documents <= 0:
            return {
                "status": "error",
                "message": "n_terms and n_documents must both be positive integers (> 0)."
            }

        # Slice matrix up to n_documents and n_terms
        doc_slice = session.feature_matrix[:n_documents, :n_terms]
        matrix_dense = doc_slice.toarray().tolist()

        terms_included = session.feature_names[:n_terms]
        doc_labels = [f"Doc {i}" for i in range(len(matrix_dense))]

        # Build figure and heatmap
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(
            matrix_dense,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=terms_included,
            yticklabels=doc_labels,
            ax=ax
        )
        ax.set_title(f"DTM Heatmap (First {len(matrix_dense)} Docs x {len(terms_included)} Terms)")
        ax.set_xlabel("Terms")
        ax.set_ylabel("Documents")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()

        # Set pending figure on session
        session.pending_figure = fig

        result_id = session.next_result_id("dtm_heatmap")
        summary = {
            "result_id": result_id,
            "status": "success",
            "n_documents": len(matrix_dense),
            "n_terms": len(terms_included),
            "terms": terms_included,
            "matrix_slice": matrix_dense
        }

        session.store_result(
            tool_name="dtm_heatmap_tool",
            args={"n_terms": n_terms, "n_documents": n_documents},
            summary=summary
        )

        return summary

    return [build_dtm_tool, term_frequency_tool, dtm_heatmap_tool]
