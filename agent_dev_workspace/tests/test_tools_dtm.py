import pytest
import numpy as np
from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from tools_dtm import make_tools


def test_build_dtm_tool_known_answer_4():
    """
    Tests build_dtm_tool against known answer #4 in TEST_FIXTURE.md using agent_dev/sample_fixture.csv.
    """
    session = SessionState()

    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")

    load_res = load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })
    assert load_res["n_documents"] == 8

    tools = make_tools(session)
    build_dtm_tool = next(t for t in tools if t.name == "build_dtm_tool")

    res = build_dtm_tool.invoke({})

    assert res["status"] == "success"
    assert res["n_documents"] == 8
    assert res["n_terms"] == 5

    # Known answer #4: vocabulary = ["alpha", "always", "beta", "delta", "gamma"]
    expected_vocab = ["alpha", "always", "beta", "delta", "gamma"]
    assert res["vocabulary"] == expected_vocab
    assert session.feature_names == expected_vocab

    # Known answer #4: non_zero = 21, total_elements = 40, sparsity_pct = 47.5
    assert res["non_zero"] == 21
    assert res["total_elements"] == 40
    assert np.isclose(res["sparsity_pct"], 47.5, atol=1e-3)

    # Check vectorizer artifact is stored
    assert "count_vectorizer" in session.artifacts
    assert session.feature_matrix is not None
    assert session.feature_matrix.shape == (8, 5)


def test_term_frequency_tool_known_answer_14():
    """
    Tests term_frequency_tool against known answer #14 in TEST_FIXTURE.md using agent_dev/sample_fixture.csv.
    """
    session = SessionState()

    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")

    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    tools = make_tools(session)
    build_dtm_tool = next(t for t in tools if t.name == "build_dtm_tool")
    term_frequency_tool = next(t for t in tools if t.name == "term_frequency_tool")

    # First build DTM
    build_dtm_tool.invoke({})

    # Compute term frequencies
    res = term_frequency_tool.invoke({})

    assert res["status"] == "success"
    assert res["total_terms"] == 5

    # Known answer #14: alpha: 6, always: 7, beta: 3, delta: 4, gamma: 7
    expected_preview = {
        "always": 7,
        "gamma": 7,
        "alpha": 6,
        "delta": 4,
        "beta": 3
    }
    assert res["top_terms_preview"] == expected_preview

    # Verify full report stored in session.results_store
    result_id = res["result_id"]
    stored_result = session.results_store[result_id]
    full_report = stored_result["full_report"]
    assert full_report is not None
    assert len(full_report) == 5
    assert dict(zip(full_report["term"], full_report["frequency"])) == expected_preview


def test_dtm_heatmap_tool_known_answer_21():
    """
    Tests dtm_heatmap_tool against known answer #21 in TEST_FIXTURE.md using agent_dev/sample_fixture.csv.
    """
    session = SessionState()

    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")

    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    tools = make_tools(session)
    build_dtm_tool = next(t for t in tools if t.name == "build_dtm_tool")
    dtm_heatmap_tool = next(t for t in tools if t.name == "dtm_heatmap_tool")

    build_dtm_tool.invoke({})

    # 1. Test defaults n_terms=20, n_documents=20 (returns full 8x5 matrix)
    res_full = dtm_heatmap_tool.invoke({"n_terms": 20, "n_documents": 20})

    assert res_full["status"] == "success"
    assert res_full["n_documents"] == 8
    assert res_full["n_terms"] == 5
    assert res_full["terms"] == ["alpha", "always", "beta", "delta", "gamma"]

    expected_full_matrix = [
        [3, 1, 1, 0, 0],
        [2, 1, 1, 0, 0],
        [1, 1, 1, 0, 0],
        [0, 1, 0, 1, 3],
        [0, 1, 0, 1, 2],
        [0, 1, 0, 1, 1],
        [0, 1, 0, 1, 1],
        [0, 0, 0, 0, 0]
    ]
    assert res_full["matrix_slice"] == expected_full_matrix
    assert session.pending_figure is not None

    # 2. Test corner slice n_terms=2, n_documents=3
    res_corner = dtm_heatmap_tool.invoke({"n_terms": 2, "n_documents": 3})

    assert res_corner["status"] == "success"
    assert res_corner["n_documents"] == 3
    assert res_corner["n_terms"] == 2
    assert res_corner["terms"] == ["alpha", "always"]
    assert res_corner["matrix_slice"] == [[3, 1], [2, 1], [1, 1]]
