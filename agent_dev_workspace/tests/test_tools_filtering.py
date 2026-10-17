"""Tests for tools in tools_filtering.py using sample_fixture.csv and known answers."""

import pytest
import numpy as np
from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from tools_filtering import make_tools


def test_variance_filter_tool_known_answer_5():
    """Test variance_filter_tool against TEST_FIXTURE.md known answer #5."""
    session = SessionState()

    # Load premade load_dataset_tool to initialize session.dataframe
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    # Build DTM to populate session.feature_matrix and session.feature_names
    from sklearn.feature_extraction.text import CountVectorizer
    vectorizer = CountVectorizer()
    session.feature_matrix = vectorizer.fit_transform(session.dataframe["text"].fillna(""))
    session.feature_names = vectorizer.get_feature_names_out().tolist()

    # Instantiate filtering tools with the same session
    filtering_tools = make_tools(session)
    variance_filter_tool = next(t for t in filtering_tools if t.name == "variance_filter_tool")

    # Call variance_filter_tool with threshold=0.15
    res = variance_filter_tool.invoke({"threshold": 0.15})

    assert res["total_features"] == 5
    assert res["kept_features_count"] == 4
    assert res["removed_features_count"] == 1

    assert set(res["kept_features"]) == {"alpha", "beta", "delta", "gamma"}
    assert set(res["removed_features"]) == {"always"}

    # Check stored report variances against known answer #5
    stored = session.results_store[res["result_id"]]
    report = stored["full_report"]
    variance_map = dict(zip(report["term"], report["variance"]))

    expected_variances = {
        "alpha": 1.1875,
        "always": 0.109375,
        "beta": 0.234375,
        "delta": 0.25,
        "gamma": 1.109375
    }

    for term, exp_var in expected_variances.items():
        assert variance_map[term] == pytest.approx(exp_var, abs=1e-3)


def test_pearson_filter_tool_known_answer_6():
    """Test pearson_filter_tool against TEST_FIXTURE.md known answer #6."""
    session = SessionState()

    # Load premade load_dataset_tool to initialize session.dataframe
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    # Build DTM to populate session.feature_matrix and session.feature_names
    from sklearn.feature_extraction.text import CountVectorizer
    vectorizer = CountVectorizer()
    session.feature_matrix = vectorizer.fit_transform(session.dataframe["text"].fillna(""))
    session.feature_names = vectorizer.get_feature_names_out().tolist()

    # Instantiate filtering tools with the same session
    filtering_tools = make_tools(session)
    pearson_filter_tool = next(t for t in filtering_tools if t.name == "pearson_filter_tool")

    # Call pearson_filter_tool with target_class="catB"
    res = pearson_filter_tool.invoke({"target_class": "catB", "threshold": 0.5})

    # Terms with |pearson_r| > 0.5: delta (1.0), gamma (0.8307), beta (-0.7746), alpha (-0.6882) -> 4 terms
    # Term removed: always (0.3780) -> 1 term
    assert res["total_features"] == 5
    assert res["kept_features_count"] == 4
    assert res["removed_features_count"] == 1

    assert set(res["kept_features"]) == {"alpha", "beta", "delta", "gamma"}
    assert set(res["removed_features"]) == {"always"}

    # Check stored report pearson_r values and row ordering (sorted by |r| descending)
    stored = session.results_store[res["result_id"]]
    report = stored["full_report"]
    assert report["term"].tolist() == ["delta", "gamma", "beta", "alpha", "always"]

    pearson_map = dict(zip(report["term"], report["pearson_r"]))

    expected_pearson = {
        "alpha": -0.6882,
        "always": 0.3780,
        "beta": -0.7746,
        "delta": 1.0000,
        "gamma": 0.8307
    }

    for term, exp_r in expected_pearson.items():
        assert pearson_map[term] == pytest.approx(exp_r, abs=1e-3)


def test_spearman_filter_tool_known_answer_6():
    """Test spearman_filter_tool against TEST_FIXTURE.md known answer #6."""
    session = SessionState()

    # Load premade load_dataset_tool to initialize session.dataframe
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    # Build DTM to populate session.feature_matrix and session.feature_names
    from sklearn.feature_extraction.text import CountVectorizer
    vectorizer = CountVectorizer()
    session.feature_matrix = vectorizer.fit_transform(session.dataframe["text"].fillna(""))
    session.feature_names = vectorizer.get_feature_names_out().tolist()

    # Instantiate filtering tools with the same session
    filtering_tools = make_tools(session)
    spearman_filter_tool = next(t for t in filtering_tools if t.name == "spearman_filter_tool")

    # Call spearman_filter_tool with target_class="catB" and threshold=0.5
    res = spearman_filter_tool.invoke({"target_class": "catB", "threshold": 0.5})

    assert res["dtm_total_features"] == 5
    assert res["evaluated_features_count"] == 5
    assert res["kept_features_count"] == 4
    assert res["removed_features_count"] == 1

    assert set(res["kept_features"]) == {"alpha", "beta", "delta", "gamma"}
    assert set(res["removed_features"]) == {"always"}

    # Check stored report spearman_r values and row ordering (sorted by |spearman_r| descending)
    stored = session.results_store[res["result_id"]]
    report = stored["full_report"]
    assert report["term"].tolist() == ["delta", "gamma", "beta", "alpha", "always"]

    spearman_map = dict(zip(report["term"], report["spearman_r"]))

    expected_spearman = {
        "alpha": -0.7500,
        "always": 0.3780,
        "beta": -0.7746,
        "delta": 1.0000,
        "gamma": 0.9363
    }

    for term, exp_r in expected_spearman.items():
        assert spearman_map[term] == pytest.approx(exp_r, abs=1e-3)


def test_spearman_filter_tool_max_terms():
    """Test spearman_filter_tool with max_terms restricting evaluated features to top 3 by variance."""
    session = SessionState()

    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    from sklearn.feature_extraction.text import CountVectorizer
    vectorizer = CountVectorizer()
    session.feature_matrix = vectorizer.fit_transform(session.dataframe["text"].fillna(""))
    session.feature_names = vectorizer.get_feature_names_out().tolist()

    filtering_tools = make_tools(session)
    spearman_filter_tool = next(t for t in filtering_tools if t.name == "spearman_filter_tool")

    # Variances: alpha (1.1875), gamma (1.109375), delta (0.25), beta (0.234375), always (0.109375)
    # Top 3 by variance: alpha, gamma, delta
    res = spearman_filter_tool.invoke({"target_class": "catB", "threshold": 0.5, "max_terms": 3})

    assert res["dtm_total_features"] == 5
    assert res["evaluated_features_count"] == 3
    assert res["max_terms_filter"] == 3

    stored = session.results_store[res["result_id"]]
    report = stored["full_report"]
    assert set(report["term"].tolist()) == {"alpha", "gamma", "delta"}
