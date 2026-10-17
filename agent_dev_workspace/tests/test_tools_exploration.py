"""Tests for tools in tools_exploration.py using sample_fixture.csv and known answers."""

import pytest
import numpy as np
from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from tools_exploration import make_tools


def test_cosine_similarity_tool_known_answer_16():
    """Test cosine_similarity_tool against TEST_FIXTURE.md known answer #16."""
    session = SessionState()

    # Load dataset using premade load_dataset_tool
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    # Build DTM on session.dataframe["text"]
    from sklearn.feature_extraction.text import CountVectorizer
    vectorizer = CountVectorizer()
    session.feature_matrix = vectorizer.fit_transform(session.dataframe["text"].fillna(""))
    session.feature_names = vectorizer.get_feature_names_out().tolist()

    # Instantiate exploration tools
    exploration_tools = make_tools(session)
    cosine_similarity_tool = next(t for t in exploration_tools if t.name == "cosine_similarity_tool")

    # Test doc 0 vs doc 1 (catA vs catA): expected 0.9847
    res_0_1 = cosine_similarity_tool.invoke({"doc_id_1": 0, "doc_id_2": 1})
    assert res_0_1["similarity"] == pytest.approx(0.9847, abs=1e-3)

    # Test doc 0 vs doc 3 (catA vs catB): expected 0.0909
    res_0_3 = cosine_similarity_tool.invoke({"doc_id_1": 0, "doc_id_2": 3})
    assert res_0_3["similarity"] == pytest.approx(0.0909, abs=1e-3)

    # Test doc 3 vs doc 4 (catB vs catB): expected 0.9847
    res_3_4 = cosine_similarity_tool.invoke({"doc_id_1": 3, "doc_id_2": 4})
    assert res_3_4["similarity"] == pytest.approx(0.9847, abs=1e-3)

    # Test doc 0 vs doc 7 (catA vs empty text row 7): expected 0.0
    res_0_7 = cosine_similarity_tool.invoke({"doc_id_1": 0, "doc_id_2": 7})
    assert res_0_7["similarity"] == 0.0


def test_feature_correlation_matrix_tool_known_answer_19():
    """Test feature_correlation_matrix_tool against TEST_FIXTURE.md known answer #19."""
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

    exploration_tools = make_tools(session)
    feature_corr_tool = next(t for t in exploration_tools if t.name == "feature_correlation_matrix_tool")

    res = feature_corr_tool.invoke({"top_n": 20})

    # Terms by variance descending: alpha, gamma, delta, beta, always
    expected_terms = ["alpha", "gamma", "delta", "beta", "always"]
    assert res["terms"] == expected_terms

    # Check pending figure is set
    assert session.pending_figure is not None

    # Expected correlation matrix matching known answer #19
    expected_matrix = np.array([
        [ 1.0000, -0.5718, -0.6882,  0.8885,  0.2601],
        [-0.5718,  1.0000,  0.8307, -0.6435,  0.3140],
        [-0.6882,  0.8307,  1.0000, -0.7746,  0.3780],
        [ 0.8885, -0.6435, -0.7746,  1.0000,  0.2928],
        [ 0.2601,  0.3140,  0.3780,  0.2928,  1.0000]
    ])

    computed_matrix = np.array(res["correlation_matrix"])
    assert np.allclose(computed_matrix, expected_matrix, atol=1e-3)
