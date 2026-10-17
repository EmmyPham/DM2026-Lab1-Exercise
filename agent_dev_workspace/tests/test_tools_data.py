import os
import pytest
import numpy as np
import pandas as pd
from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from tools_data import make_tools


def test_inspect_data_tool():
    """
    Tests inspect_data_tool using agent_dev/sample_fixture.csv.
    Verifies shape, column names, dtypes, n_rows_previewed, and head row preview
    for positive n (verifying exact top 5 text content), negative n (verifying exact tail content), and n=0.
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
    inspect_data_tool = next(t for t in tools if t.name == "inspect_data_tool")

    # 1. Test default n=5 (verifying exact text content of top 5 rows)
    res = inspect_data_tool.invoke({"n": 5})
    assert res["status"] == "success"
    assert res["shape"] == [8, 3]  # text, category_name, category
    assert "text" in res["columns"]
    assert "category_name" in res["columns"]
    assert "category" in res["columns"]
    assert res["dtypes"]["text"] == "object"
    assert res["n_rows_previewed"] == 5
    assert len(res["head"]) == 5

    head_texts = [row["text"] for row in res["head"]]
    assert head_texts == [
        "always alpha alpha alpha beta",
        "always alpha alpha beta",
        "always alpha beta",
        "always gamma gamma gamma delta",
        "always gamma gamma delta"
    ]

    # 2. Test negative n=-3 (tail preview: asserts exact text values of bottom 3 rows)
    res_tail = inspect_data_tool.invoke({"n": -3})
    assert res_tail["status"] == "success"
    assert res_tail["n_rows_previewed"] == 3
    assert len(res_tail["head"]) == 3
    
    # Verify the actual content of the last 3 rows (indices 5, 6, 7)
    tail_texts = [row["text"] for row in res_tail["head"]]
    assert tail_texts == ["always gamma delta", "always gamma delta", ""]

    # 3. Test n=0
    res_zero = inspect_data_tool.invoke({"n": 0})
    assert res_zero["status"] == "success"
    assert res_zero["n_rows_previewed"] == 0
    assert len(res_zero["head"]) == 0


def test_tokenize_tool_known_answer_10():
    """
    Tests tokenize_tool against known answer #10 in TEST_FIXTURE.md using agent_dev/sample_fixture.csv.
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
    tokenize_tool = next(t for t in tools if t.name == "tokenize_tool")

    result = tokenize_tool.invoke({})

    expected_tokens = [
        ["always", "alpha", "alpha", "alpha", "beta"],
        ["always", "alpha", "alpha", "beta"],
        ["always", "alpha", "beta"],
        ["always", "gamma", "gamma", "gamma", "delta"],
        ["always", "gamma", "gamma", "delta"],
        ["always", "gamma", "delta"],
        ["always", "gamma", "delta"],
        []
    ]

    unigrams_in_df = session.dataframe["unigrams"].tolist()
    assert unigrams_in_df == expected_tokens

    unigrams_in_artifacts = session.artifacts["unigrams"]
    assert unigrams_in_artifacts == expected_tokens

    total_expected_tokens = sum(len(t) for t in expected_tokens)
    avg_expected_tokens = total_expected_tokens / len(expected_tokens)

    assert result["status"] == "success"
    assert result["num_documents"] == 8
    assert result["total_tokens"] == total_expected_tokens
    assert pytest.approx(result["avg_tokens_per_doc"]) == avg_expected_tokens
    assert result["sample_tokens"] == ["always", "alpha", "alpha", "alpha", "beta"]


def test_list_files_tool_known_answer_11():
    """
    Tests list_files_tool against known answer #11 in TEST_FIXTURE.md for subdirectory="newdataset",
    and verifies that paths escaping the repo root return an error status.
    """
    session = SessionState()
    tools = make_tools(session)
    list_files_tool = next(t for t in tools if t.name == "list_files_tool")

    # Known answer #11 check
    res = list_files_tool.invoke({"subdirectory": "newdataset"})
    assert res["status"] == "success"
    assert res["subdirectory"] == "newdataset"
    assert res["directories"] == []
    assert res["files"] == ["Reddit-stock-sentiment.csv"]

    # Security check: path outside repo
    res_outside = list_files_tool.invoke({"subdirectory": "../.."})
    assert res_outside["status"] == "error"
    assert "resolves outside repository root" in res_outside["message"]


def test_check_missing_tool_known_answer_2():
    """
    Tests check_missing_tool against known answer #2 in TEST_FIXTURE.md using agent_dev/sample_fixture.csv.
    Computes expected missing rows dynamically with pandas and asserts tool output matches.
    """
    session = SessionState()

    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")

    fixture_path = "agent_dev/sample_fixture.csv"
    load_res = load_dataset_tool.invoke({
        "file_path": fixture_path,
        "text_column": "text",
        "label_column": "label"
    })
    assert load_res["n_documents"] == 8

    tools = make_tools(session)
    check_missing_tool = next(t for t in tools if t.name == "check_missing_tool")

    result = check_missing_tool.invoke({})

    # Compute expected values independently using pandas directly on the CSV file
    df_raw = pd.read_csv(fixture_path)
    expected_missing_indices = [
        idx for idx, val in enumerate(df_raw["text"])
        if pd.isna(val) or not str(val).strip()
    ]
    expected_missing_count = len(expected_missing_indices)
    expected_missing_pct = (expected_missing_count / len(df_raw)) * 100

    # Assertions comparing tool results to independent pandas computation
    assert result["status"] == "success"
    assert result["num_documents"] == len(df_raw)
    assert result["missing_count"] == expected_missing_count
    assert result["missing_indices"] == expected_missing_indices
    assert pytest.approx(result["missing_pct"], abs=1e-3) == expected_missing_pct


def test_check_duplicates_tool_known_answer_3():
    """
    Tests check_duplicates_tool against known answer #3 in TEST_FIXTURE.md using agent_dev/sample_fixture.csv.
    """
    # 1. Inspect duplicates without dropping (drop=False)
    session = SessionState()
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })
    
    tools = make_tools(session)
    check_duplicates_tool = next(t for t in tools if t.name == "check_duplicates_tool")
    
    dup_res = check_duplicates_tool.invoke({"columns": None, "drop": False})
    
    assert dup_res["status"] == "success"
    assert dup_res["n_initial"] == 8
    assert dup_res["n_duplicates"] == 1
    assert dup_res["duplicate_indices"] == [6]
    assert dup_res["n_remaining"] == 8
    assert len(session.dataframe) == 8

    # 2. Drop duplicates with keep=False (drop=True)
    session_drop = SessionState()
    premade_tools_drop, _ = load_workspace_tools("premade_tools", session_drop)
    load_dataset_tool_drop = next(t for t in premade_tools_drop if t.name == "load_dataset_tool")
    
    load_dataset_tool_drop.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })
    
    tools_drop = make_tools(session_drop)
    check_duplicates_tool_drop = next(t for t in tools_drop if t.name == "check_duplicates_tool")
    
    drop_res = check_duplicates_tool_drop.invoke({"columns": None, "drop": True})
    
    assert drop_res["status"] == "success"
    assert drop_res["n_initial"] == 8
    assert drop_res["n_duplicates"] == 1
    assert drop_res["dropped"] is True
    assert drop_res["n_remaining"] == 6
    assert len(session_drop.dataframe) == 6


def test_sample_data_tool_known_answer_13():
    """
    Tests sample_data_tool against known answer #13 in TEST_FIXTURE.md using agent_dev/sample_fixture.csv.
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
    sample_data_tool = next(t for t in tools if t.name == "sample_data_tool")

    # Known Answer #13: Call with n=4, random_state=42 returns exactly rows [1, 5, 0, 7]
    res = sample_data_tool.invoke({"n": 4, "random_state": 42})

    assert res["status"] == "success"
    assert res["n_initial"] == 8
    assert res["n_sampled"] == 4
    assert res["sampled_indices"] == [1, 5, 0, 7]
    assert len(session.dataframe) == 4


def test_describe_data_tool_known_answer_18():
    """
    Tests describe_data_tool against known answer #18 in TEST_FIXTURE.md using agent_dev/sample_fixture.csv.
    """
    session = SessionState()

    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")

    load_res = load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })
    assert load_res.get("n_documents") == 8

    tools = make_tools(session)
    describe_data_tool = next(t for t in tools if t.name == "describe_data_tool")

    res = describe_data_tool.invoke({})

    # Overall stats assertions against known answer #18:
    overall = res["overall_stats"]
    assert overall["count"] == 8
    assert np.isclose(overall["mean"], 19.875, atol=1e-3)
    assert np.isclose(overall["std"], 9.4330, atol=1e-3)
    assert overall["min"] == 0
    assert np.isclose(overall["25%"], 17.75, atol=1e-3)
    assert np.isclose(overall["50%"], 20.5, atol=1e-3)
    assert np.isclose(overall["75%"], 25.25, atol=1e-3)
    assert overall["max"] == 30

    # Per category stats assertions:
    per_cat = res["per_category_stats"]
    assert "catA" in per_cat
    catA = per_cat["catA"]
    assert catA["count"] == 4
    assert np.isclose(catA["mean"], 17.25, atol=1e-3)
    assert np.isclose(catA["std"], 12.5, atol=1e-3)
    assert catA["min"] == 0
    assert np.isclose(catA["25%"], 12.75, atol=1e-3)
    assert np.isclose(catA["50%"], 20.0, atol=1e-3)
    assert np.isclose(catA["75%"], 24.5, atol=1e-3)
    assert catA["max"] == 29

    assert "catB" in per_cat
    catB = per_cat["catB"]
    assert catB["count"] == 4
    assert np.isclose(catB["mean"], 22.5, atol=1e-3)
    assert np.isclose(catB["std"], 5.7446, atol=1e-3)
    assert catB["min"] == 18
    assert np.isclose(catB["25%"], 18.0, atol=1e-3)
    assert np.isclose(catB["50%"], 21.0, atol=1e-3)
    assert np.isclose(catB["75%"], 25.5, atol=1e-3)
    assert catB["max"] == 30

    # Assert that session.pending_figure is produced
    assert session.pending_figure is not None
