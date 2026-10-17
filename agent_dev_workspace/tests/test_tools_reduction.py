"""Tests for tools in tools_reduction.py using sample_fixture.csv and known answers."""

import pytest
import numpy as np
from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from tools_reduction import make_tools


def test_reduce_dimensions_tool_pca_known_answer_7():
    """Test reduce_dimensions_tool method='pca' against TEST_FIXTURE.md known answer #7 across all 8 rows."""
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

    reduction_tools = make_tools(session)
    reduce_dimensions_tool = next(t for t in reduction_tools if t.name == "reduce_dimensions_tool")

    res = reduce_dimensions_tool.invoke({"method": "pca", "n_components": 2})

    # Check explained variance ratio
    assert res["explained_variance_ratio"] == pytest.approx([0.7532, 0.1965], abs=1e-3)

    # Check all 8 rows from known answer #7
    expected_pca = np.array([
        [2.3669, 0.8871],
        [1.7039, 0.2642],
        [1.0408, -0.3587],
        [-2.0759, 1.0129],
        [-1.4554, 0.3405],
        [-0.8349, -0.3319],
        [-0.8349, -0.3319],
        [0.0894, -1.4822]
    ])

    coords = session.artifacts["reduced_pca"]
    assert np.allclose(coords, expected_pca, atol=1e-3)
    assert session.pending_figure is not None


def test_reduce_dimensions_tool_tsne_structural():
    """Test reduce_dimensions_tool method='tsne' structurally (shape, finiteness, reproducibility, duplicate rows)."""
    session1 = SessionState()

    premade_tools, _ = load_workspace_tools("premade_tools", session1)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    from sklearn.feature_extraction.text import CountVectorizer
    vectorizer = CountVectorizer()
    session1.feature_matrix = vectorizer.fit_transform(session1.dataframe["text"].fillna(""))
    session1.feature_names = vectorizer.get_feature_names_out().tolist()

    reduction_tools1 = make_tools(session1)
    reduce_dimensions_tool1 = next(t for t in reduction_tools1 if t.name == "reduce_dimensions_tool")

    res1 = reduce_dimensions_tool1.invoke({
        "method": "tsne",
        "n_components": 2,
        "perplexity": 2,
        "random_state": 42
    })

    coords1 = session1.artifacts["reduced_tsne"]

    # 1. Output shape is (8, 2)
    assert coords1.shape == (8, 2)

    # 2. All values are finite
    assert np.all(np.isfinite(coords1))

    # 3. Duplicate rows (row 5 and row 6, index 5 and index 6) have identical coordinates
    assert np.allclose(coords1[5], coords1[6], atol=1e-4)

    # 4. Reproducibility: second run with same random_state gives identical output
    session2 = SessionState()
    load_dataset_tool2 = next(t for t in load_workspace_tools("premade_tools", session2)[0] if t.name == "load_dataset_tool")
    load_dataset_tool2.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })
    session2.feature_matrix = vectorizer.fit_transform(session2.dataframe["text"].fillna(""))
    session2.feature_names = vectorizer.get_feature_names_out().tolist()

    reduction_tools2 = make_tools(session2)
    reduce_dimensions_tool2 = next(t for t in reduction_tools2 if t.name == "reduce_dimensions_tool")

    res2 = reduce_dimensions_tool2.invoke({
        "method": "tsne",
        "n_components": 2,
        "perplexity": 2,
        "random_state": 42
    })

    coords2 = session2.artifacts["reduced_tsne"]
    assert np.allclose(coords1, coords2, atol=1e-4)
    assert session1.pending_figure is not None


def test_reduce_dimensions_tool_umap_known_answer_9():
    """Test reduce_dimensions_tool method='umap' against TEST_FIXTURE.md known answer #9 across all 8 rows."""
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

    reduction_tools = make_tools(session)
    reduce_dimensions_tool = next(t for t in reduction_tools if t.name == "reduce_dimensions_tool")

    res = reduce_dimensions_tool.invoke({
        "method": "umap",
        "n_components": 2,
        "n_neighbors": 3,
        "random_state": 42
    })

    # Check all 8 rows from known answer #9
    expected_umap = np.array([
        [6.6811, -6.8218],
        [6.9907, -7.2245],
        [7.7022, -7.3712],
        [10.2228, -6.2310],
        [9.8759, -5.6537],
        [9.3831, -6.5748],
        [9.2153, -5.8297],
        [8.5860, -7.2622]
    ])

    coords = session.artifacts["reduced_umap"]
    assert np.allclose(coords, expected_umap, atol=1e-1)
    assert session.pending_figure is not None


def test_binarize_labels_tool():
    """Test binarize_labels_tool using sample_fixture.csv."""
    session = SessionState()

    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    reduction_tools = make_tools(session)
    binarize_labels_tool = next(t for t in reduction_tools if t.name == "binarize_labels_tool")

    res = binarize_labels_tool.invoke({})

    assert res["n_samples"] == 8
    assert res["n_classes"] == 2
    assert res["classes"] == ["catA", "catB"]

    # Verify one-hot matrix shape (8, 2)
    binarized_matrix = session.artifacts["binarized_labels"]
    assert binarized_matrix.shape == (8, 2)

    # First 4 rows are catA ([1, 0]), next 4 rows are catB ([0, 1])
    expected_matrix = np.array([
        [1, 0],
        [1, 0],
        [1, 0],
        [0, 1],
        [0, 1],
        [0, 1],
        [0, 1],
        [1, 0]  # row 7 is catA
    ])
    assert np.array_equal(binarized_matrix, expected_matrix)
