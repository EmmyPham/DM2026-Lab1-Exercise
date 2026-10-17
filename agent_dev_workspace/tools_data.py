import os
import re
from pathlib import Path
import pandas as pd
import numpy as np
import nltk
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Optional, Union
from nltk.tokenize import word_tokenize
from langchain_core.tools import tool

# Ensure punkt tokenizer models are available
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab', quiet=True)


def make_tools(session):
    @tool
    def inspect_data_tool(n: int = 5) -> dict:
        """
        Inspects the active dataset in session.dataframe, returning its shape, column names,
        data types, warnings, and a preview of rows.

        Args:
            n (int): Number of rows to preview. If n > 0, returns top n rows (head).
                If n < 0, returns bottom |n| rows (tail). If n == 0, returns empty preview.
                Default is 5.

        Returns:
            dict: Summary containing result_id, status, shape, columns, dtypes,
                  preview_rows, head, and warnings.
        """
        if session.dataframe is None:
            return {
                "status": "error",
                "message": "No dataset loaded in session. Please load a dataset first."
            }

        df = session.dataframe
        warnings = []

        if df.empty:
            warnings.append("Active dataset is empty (0 rows).")

        # Check for expected standard pipeline columns
        expected_cols = ["text", "category_name"]
        for col in expected_cols:
            if col not in df.columns:
                warnings.append(f"Column '{col}' is missing from dataframe.")

        shape = [len(df), len(df.columns)]
        columns = list(df.columns)
        dtypes = {col: str(df[col].dtype) for col in df.columns}

        # Handle row selection logic
        if n > 0:
            preview_df = df.head(n)
        elif n < 0:
            preview_df = df.tail(abs(n))
        else:
            preview_df = df.head(0)

        # Convert preview to records (JSON-serializable)
        head_records = preview_df.to_dict(orient="records")

        # Convert any numpy/pandas types in records for clean JSON serialization
        for row in head_records:
            for k, v in row.items():
                if pd.isna(v):
                    row[k] = None
                elif isinstance(v, (np.integer, np.int64, np.int32)):
                    row[k] = int(v)
                elif isinstance(v, (np.floating, np.float64, np.float32)):
                    row[k] = float(v)

        result_id = session.next_result_id("inspect_data")
        summary = {
            "result_id": result_id,
            "status": "success",
            "shape": shape,
            "columns": columns,
            "dtypes": dtypes,
            "n_rows_previewed": len(preview_df),
            "head": head_records
        }

        if warnings:
            summary["warnings"] = warnings

        session.store_result("inspect_data_tool", {"n": n}, summary)
        return summary

    @tool
    def tokenize_tool() -> dict:
        """
        Tokenizes document text in the active dataset into lowercased unigrams using NLTK word_tokenize.
        
        Reads session.dataframe['text'], tokenizes each document into lowercased words,
        stores the result in session.dataframe['unigrams'] and session.artifacts['unigrams'],
        and returns summary statistics.
        
        Returns:
            dict: Summary containing result_id, status, num_documents, total_tokens,
                  avg_tokens_per_doc, and sample_tokens preview.
        """
        if session.dataframe is None or session.dataframe.empty:
            return {
                "status": "error",
                "message": "No dataframe found in session. Please load a dataset first."
            }

        if "text" not in session.dataframe.columns:
            return {
                "status": "error",
                "message": "Dataframe missing required 'text' column."
            }

        tokens_list = []
        for text in session.dataframe["text"]:
            if pd.isna(text) or not str(text).strip():
                tokens_list.append([])
            else:
                tokens = word_tokenize(str(text).lower())
                tokens_list.append(tokens)

        # Update session state
        session.dataframe["unigrams"] = tokens_list
        session.artifacts["unigrams"] = tokens_list

        total_tokens = sum(len(t) for t in tokens_list)
        num_docs = len(tokens_list)
        avg_tokens = float(total_tokens / num_docs) if num_docs > 0 else 0.0

        sample_preview = tokens_list[0][:5] if num_docs > 0 and len(tokens_list[0]) > 0 else []

        result_id = session.next_result_id("tokenize")
        summary = {
            "result_id": result_id,
            "status": "success",
            "num_documents": num_docs,
            "total_tokens": total_tokens,
            "avg_tokens_per_doc": round(avg_tokens, 4),
            "sample_tokens": sample_preview
        }

        session.store_result("tokenize_tool", {}, summary)
        return summary

    @tool
    def list_files_tool(subdirectory: str = "") -> dict:
        """
        Lists files and subdirectories in the repo or a given subdirectory within the repository boundary.

        Args:
            subdirectory (str): Subdirectory path relative to the workspace/repo root. Default is "" (root).

        Returns:
            dict: Summary containing result_id, status, subdirectory, directories, and files.
        """
        repo_root = Path(".").resolve()
        target_path = (repo_root / subdirectory).resolve() if subdirectory else repo_root

        # Ensure target_path stays strictly inside repo_root
        try:
            target_path.relative_to(repo_root)
        except ValueError:
            return {
                "status": "error",
                "message": f"Access denied: path '{subdirectory}' resolves outside repository root."
            }

        if not target_path.exists():
            return {
                "status": "error",
                "message": f"Path '{subdirectory}' does not exist."
            }

        if target_path.is_file():
            directories = []
            files = [target_path.name]
        else:
            directories = []
            files = []
            for item in target_path.iterdir():
                if item.is_dir():
                    directories.append(item.name)
                elif item.is_file():
                    files.append(item.name)

            directories.sort()
            files.sort()

        result_id = session.next_result_id("list_files")
        summary = {
            "result_id": result_id,
            "status": "success",
            "subdirectory": subdirectory,
            "directories": directories,
            "files": files
        }

        session.store_result(
            tool_name="list_files_tool",
            args={"subdirectory": subdirectory},
            summary=summary
        )

        return summary

    @tool
    def check_missing_tool(columns: Optional[Union[str, List[str]]] = None) -> dict:
        """
        Checks for missing or empty values in specified column(s) of the active dataset.
        
        Inspects specified columns (or 'text' by default) for NaN values or empty/whitespace-only strings,
        reporting missing counts, percentages, and row indices per column and overall.
        
        Args:
            columns (Optional[Union[str, List[str]]]): Name of column or list of column names to inspect.
                If None, defaults to checking ['text'].
        
        Returns:
            dict: Summary containing result_id, status, num_documents, missing_by_column,
                  and rows_with_any_missing_count.
        """
        if session.dataframe is None:
            return {
                "status": "error",
                "message": "No dataset loaded in session. Please load a dataset first."
            }

        if session.dataframe.empty:
            return {
                "status": "error",
                "message": "Active dataset is empty (0 rows)."
            }

        # Normalize input columns into a list
        if columns is None:
            cols_to_check = ["text"] if "text" in session.dataframe.columns else list(session.dataframe.columns)
        elif isinstance(columns, str):
            cols_to_check = [columns]
        else:
            cols_to_check = list(columns)

        # Validate that requested columns exist
        missing_cols = [c for c in cols_to_check if c not in session.dataframe.columns]
        if missing_cols:
            return {
                "status": "error",
                "message": f"Column(s) {missing_cols} not found in dataframe. Available columns: {list(session.dataframe.columns)}"
            }

        num_docs = len(session.dataframe)
        missing_by_column = {}
        rows_with_any_missing = set()

        for col in cols_to_check:
            missing_indices = []
            for idx, val in enumerate(session.dataframe[col]):
                if pd.isna(val) or not str(val).strip():
                    missing_indices.append(idx)
                    rows_with_any_missing.add(idx)

            col_missing_count = len(missing_indices)
            col_missing_pct = float(col_missing_count / num_docs * 100) if num_docs > 0 else 0.0

            missing_by_column[col] = {
                "missing_count": col_missing_count,
                "missing_indices": missing_indices,
                "missing_pct": round(col_missing_pct, 4)
            }

        # For single column input, also provide top-level summary keys for backward compatibility / fixture testing
        first_col_stats = missing_by_column[cols_to_check[0]]
        
        result_id = session.next_result_id("check_missing")
        summary = {
            "result_id": result_id,
            "status": "success",
            "num_documents": num_docs,
            "missing_count": first_col_stats["missing_count"],
            "missing_indices": first_col_stats["missing_indices"],
            "missing_pct": first_col_stats["missing_pct"],
            "missing_by_column": missing_by_column,
            "rows_with_any_missing_count": len(rows_with_any_missing)
        }

        session.store_result("check_missing_tool", {"columns": columns}, summary)
        return summary

    @tool
    def check_duplicates_tool(
        columns: Optional[Union[str, List[str]]] = None,
        drop: bool = False
    ) -> dict:
        """
        Checks for duplicate documents in the active dataset, and optionally drops them.
        
        When drop=True, duplicate rows are removed using keep=False (dropping every copy
        of a duplicated row, matching Master's drop_duplicates(keep=False) specification).
        
        Args:
            columns (Optional[Union[str, List[str]]]): Column name or list of column names
                to check for duplicates. If None, checks across all columns.
            drop (bool): Whether to drop duplicate rows from session.dataframe. Default is False.
        
        Returns:
            dict: Summary containing result_id, status, n_initial, n_duplicates,
                  duplicate_indices, dropped, and n_remaining.
        """
        if session.dataframe is None:
            return {
                "status": "error",
                "message": "No dataset loaded in session. Please load a dataset first."
            }

        if session.dataframe.empty:
            return {
                "status": "error",
                "message": "Active dataset is empty (0 rows)."
            }

        if columns is None:
            cols_to_check = None
        elif isinstance(columns, str):
            cols_to_check = [columns]
        else:
            cols_to_check = list(columns)

        if cols_to_check is not None:
            missing_cols = [c for c in cols_to_check if c not in session.dataframe.columns]
            if missing_cols:
                return {
                    "status": "error",
                    "message": f"Column(s) {missing_cols} not found in dataframe. Available columns: {list(session.dataframe.columns)}"
                }

        n_initial = len(session.dataframe)

        # First identify duplicate mask with keep='first' or keep=False
        dup_mask = session.dataframe.duplicated(subset=cols_to_check, keep="first")
        duplicate_indices = list(np.where(dup_mask)[0])
        n_duplicates = len(duplicate_indices)

        if drop and n_duplicates > 0:
            session.dataframe = session.dataframe.drop_duplicates(
                subset=cols_to_check, keep=False
            ).reset_index(drop=True)

        n_remaining = len(session.dataframe)

        result_id = session.next_result_id("check_duplicates")
        summary = {
            "result_id": result_id,
            "status": "success",
            "n_initial": n_initial,
            "n_duplicates": n_duplicates,
            "duplicate_indices": duplicate_indices,
            "dropped": drop,
            "n_remaining": n_remaining
        }

        session.store_result("check_duplicates_tool", {"columns": columns, "drop": drop}, summary)
        return summary

    @tool
    def sample_data_tool(
        n: Optional[int] = None,
        frac: Optional[float] = None,
        random_state: Optional[int] = None,
        replace: bool = False
    ) -> dict:
        """
        Subsamples the working DataFrame in session.dataframe.

        Args:
            n (Optional[int]): Number of items to sample. Cannot be combined with frac.
            frac (Optional[float]): Fraction of items to sample. Cannot be combined with n.
            random_state (Optional[int]): Seed for the random number generator for reproducibility.
            replace (bool): Allow or disallow sampling of the same row more than once. Default is False.

        Returns:
            dict: Summary containing result_id, status, n_initial, n_sampled, and sampled_indices.
        """
        if session.dataframe is None or session.dataframe.empty:
            return {
                "status": "error",
                "message": "No dataframe found in session to sample from."
            }

        if n is not None and frac is not None:
            return {
                "status": "error",
                "message": "Cannot specify both 'n' and 'frac'. Please pass either 'n' or 'frac'."
            }

        if n is None and frac is None:
            return {
                "status": "error",
                "message": "Please specify either 'n' (number of rows) or 'frac' (fraction of rows) to sample."
            }

        n_initial = len(session.dataframe)

        if n is not None and not replace and n > n_initial:
            return {
                "status": "error",
                "message": f"Sample size n={n} is larger than dataframe size ({n_initial}). Use replace=True to allow sampling with replacement."
            }

        if frac is not None and not replace and frac > 1.0:
            return {
                "status": "error",
                "message": f"Sample fraction frac={frac} > 1.0 is larger than dataframe size ({n_initial}). Use replace=True to allow sampling with replacement."
            }

        # Perform sampling
        sampled_df = session.dataframe.sample(
            n=n, frac=frac, replace=replace, random_state=random_state
        )
        sampled_indices = sampled_df.index.tolist()

        # Update session.dataframe with sampled subset and reset index cleanly
        session.dataframe = sampled_df.reset_index(drop=True)

        result_id = session.next_result_id("sample_data")
        summary = {
            "result_id": result_id,
            "status": "success",
            "n_initial": n_initial,
            "n_sampled": len(sampled_df),
            "sampled_indices": sampled_indices
        }

        session.store_result(
            tool_name="sample_data_tool",
            args={"n": n, "frac": frac, "random_state": random_state, "replace": replace},
            summary=summary
        )

        return summary

    @tool
    def describe_data_tool() -> dict:
        """Compute text length statistics (overall and per category) and produce a box plot.

        Returns:
            dict: Summary containing overall and per-category statistics, warnings if missing
            values or columns are present, and result_id.
        """
        if session.dataframe is None:
            return {"status": "error", "message": "No dataframe loaded in session."}

        df = session.dataframe.copy()
        if "text" not in df.columns:
            return {"status": "error", "message": "DataFrame must contain a 'text' column."}

        warnings = []

        # Check for missing/null text values
        missing_count = df["text"].isna().sum()
        if missing_count > 0:
            warnings.append(
                f"DataFrame contains {missing_count} missing text value(s); treated as length 0."
            )

        # Compute character length (treating NaN/None as empty string "")
        df["text_length"] = df["text"].fillna("").astype(str).apply(len)

        # Overall summary statistics
        overall_stats = df["text_length"].describe().to_dict()

        # Per-category summary statistics & warning if missing category_name
        per_category_stats = {}
        if "category_name" in df.columns:
            for cat, group in df.groupby("category_name"):
                per_category_stats[str(cat)] = group["text_length"].describe().to_dict()
        else:
            warnings.append(
                "Column 'category_name' not found in DataFrame; per-category statistics omitted."
            )

        # Generate box plot
        fig, ax = plt.subplots(figsize=(8, 5))
        if "category_name" in df.columns:
            sns.boxplot(data=df, x="category_name", y="text_length", ax=ax)
            ax.set_title("Text Length Distribution by Category")
            ax.set_xlabel("Category")
        else:
            sns.boxplot(y=df["text_length"], ax=ax)
            ax.set_title("Text Length Distribution")
        ax.set_ylabel("Text Length (characters)")
        plt.tight_layout()

        # Assign figure to session.pending_figure for rendering/saving
        session.pending_figure = fig

        result_id = session.next_result_id("describe_data")
        summary = {
            "result_id": result_id,
            "overall_stats": overall_stats,
            "per_category_stats": per_category_stats,
        }
        if warnings:
            summary["warnings"] = warnings

        session.store_result(
            tool_name="describe_data_tool",
            args={},
            summary=summary,
        )

        return summary

    return [
        inspect_data_tool,
        tokenize_tool,
        list_files_tool,
        check_missing_tool,
        check_duplicates_tool,
        sample_data_tool,
        describe_data_tool
    ]
