"""Tools for frequent pattern mining using PAMI."""

import tempfile
import os
from typing import Literal, Optional
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

from langchain_core.tools import tool


def make_tools(session):
    @tool
    def mine_patterns_tool(
        category_name: str,
        filtering_method: Literal["variance", "tfidf", "term_frequency"] = "tfidf",
        algorithm: Literal["fpgrowth", "topk", "maxfpgrowth"] = "fpgrowth",
        min_sup: Optional[int] = 1,
        k: Optional[int] = 10
    ) -> dict:
        """Mine frequent itemsets/patterns for a specific category using PAMI.

        Args:
            category_name: Category to filter documents by before mining.
            filtering_method: Vocabulary filtering method ('variance', 'tfidf', or 'term_frequency').
            algorithm: PAMI algorithm ('fpgrowth', 'topk', or 'maxfpgrowth').
            min_sup: Minimum support threshold for fpgrowth and maxfpgrowth.
            k: Top-k patterns count for topk algorithm.

        Returns:
            dict: Summary containing mined patterns, support values, and category metadata.
        """
        if session.dataframe is None or session.dataframe.empty:
            return {"error": "No DataFrame found in session. Load data first."}

        # Filter session.dataframe to target category
        if "category_name" in session.dataframe.columns:
            cat_df = session.dataframe[session.dataframe["category_name"] == category_name]
        elif "category" in session.dataframe.columns:
            cat_df = session.dataframe[session.dataframe["category"] == category_name]
        else:
            return {"error": "Category column not found in session.dataframe."}

        if cat_df.empty:
            return {"error": f"No documents found for category '{category_name}'."}

        # Step 1: Per-category local DTM with English stopwords removed
        texts = cat_df["text"].fillna("").tolist()
        vectorizer = CountVectorizer(stop_words="english")
        try:
            dtm = vectorizer.fit_transform(texts).toarray()
            feature_names = np.array(vectorizer.get_feature_names_out())
        except ValueError:
            # Handles empty vocabulary case
            return {
                "result_id": session.next_result_id("patterns"),
                "category_name": category_name,
                "patterns": [],
                "note": "no terms survived tokenization"
            }

        if dtm.shape[1] == 0:
            return {
                "result_id": session.next_result_id("patterns"),
                "category_name": category_name,
                "patterns": [],
                "note": "no terms survived tokenization"
            }

        # Step 2: Apply chosen filtering_method
        if filtering_method == "variance":
            variances = np.var(dtm, axis=0)
            p5, p95 = np.percentile(variances, 5), np.percentile(variances, 95)
            keep_mask = (variances >= p5) & (variances <= p95)

        elif filtering_method == "term_frequency":
            counts = np.sum(dtm, axis=0)
            p5, p95 = np.percentile(counts, 5), np.percentile(counts, 95)
            keep_mask = (counts >= p5) & (counts <= p95)

        elif filtering_method == "tfidf":
            tfidf_mat = TfidfTransformer().fit_transform(dtm).toarray()
            mean_tfidf = np.mean(tfidf_mat, axis=0)
            p20 = np.percentile(mean_tfidf, 20)
            keep_mask = mean_tfidf >= p20

        else:
            return {"error": f"Unknown filtering method '{filtering_method}'."}

        kept_terms = feature_names[keep_mask]
        kept_dtm = dtm[:, keep_mask]

        # Step 3: Handle zero surviving terms cleanly
        if len(kept_terms) == 0 or kept_dtm.shape[1] == 0:
            result_id = session.next_result_id("patterns")
            summary = {
                "result_id": result_id,
                "category_name": category_name,
                "filtering_method": filtering_method,
                "algorithm": algorithm,
                "patterns": [],
                "note": "no terms survived filtering"
            }
            artifact_key = f"patterns_{category_name}_{filtering_method}_{algorithm}"
            session.artifacts[artifact_key] = []
            session.store_result("mine_patterns_tool", {"category_name": category_name}, summary)
            return summary

        # Step 4: Convert filtered matrix into transactional database via PAMI DF2DB helper
        binary_df = pd.DataFrame(kept_dtm, columns=kept_terms)
        binary_df = binary_df.loc[(binary_df > 0).any(axis=1)]

        if binary_df.empty:
            result_id = session.next_result_id("patterns")
            summary = {
                "result_id": result_id,
                "category_name": category_name,
                "filtering_method": filtering_method,
                "algorithm": algorithm,
                "patterns": [],
                "note": "no transactions formed after filtering"
            }
            artifact_key = f"patterns_{category_name}_{filtering_method}_{algorithm}"
            session.artifacts[artifact_key] = []
            session.store_result("mine_patterns_tool", {"category_name": category_name}, summary)
            return summary

        temp_db_file = tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt")
        temp_db_file.close()

        try:
            from PAMI.extras.convert import DF2DB
            converter = DF2DB.DF2DB(binary_df)
            converter.convert2TransactionalDatabase(temp_db_file.name, ">=", 1)

            # Mine with requested PAMI algorithm
            if algorithm == "fpgrowth":
                from PAMI.frequentPattern.basic import FPGrowth
                obj = FPGrowth.FPGrowth(temp_db_file.name, minSup=min_sup, sep="\t")
            elif algorithm == "topk":
                from PAMI.frequentPattern.topk import FAE
                obj = FAE.FAE(temp_db_file.name, k=k, sep="\t")
            elif algorithm == "maxfpgrowth":
                from PAMI.frequentPattern.maximal import MaxFPGrowth
                obj = MaxFPGrowth.MaxFPGrowth(temp_db_file.name, minSup=min_sup, sep="\t")
            else:
                return {"error": f"Unknown algorithm '{algorithm}'."}

            obj.startMine()
            pami_df = obj.getPatternsAsDataFrame()

            patterns = []
            if pami_df is not None and not pami_df.empty:
                for _, row in pami_df.iterrows():
                    pattern_str = str(row["Patterns"])
                    supp = int(row["Support"])
                    patterns.append({
                        "pattern": pattern_str.split("\t") if "\t" in pattern_str else pattern_str.split(),
                        "support": supp
                    })

        finally:
            if os.path.exists(temp_db_file.name):
                os.remove(temp_db_file.name)

        artifact_key = f"patterns_{category_name}_{filtering_method}_{algorithm}"
        session.artifacts[artifact_key] = patterns

        result_id = session.next_result_id("patterns")
        summary = {
            "result_id": result_id,
            "category_name": category_name,
            "filtering_method": filtering_method,
            "algorithm": algorithm,
            "total_patterns": len(patterns),
            "patterns": patterns[:20]
        }
        session.store_result("mine_patterns_tool", {"category_name": category_name, "algorithm": algorithm}, summary)
        return summary

    return [mine_patterns_tool]
