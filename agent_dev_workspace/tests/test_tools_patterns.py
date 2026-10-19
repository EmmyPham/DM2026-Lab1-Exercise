"""Tests for mine_patterns_tool in tools_patterns.py against TEST_FIXTURE.md known answers #17 and #20."""

import pytest
import pandas as pd
from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from tools_patterns import make_tools


def test_mine_patterns_fpgrowth_known_answer_17():
    """Test mine_patterns_tool with fpgrowth against TEST_FIXTURE.md known answer #17."""
    session = SessionState()

    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    pattern_tools = make_tools(session)
    mine_patterns_tool = next(t for t in pattern_tools if t.name == "mine_patterns_tool")

    # 1. Variance filtering keeps 0 terms on this fixture
    res_var = mine_patterns_tool.invoke({
        "category_name": "catA",
        "filtering_method": "variance",
        "algorithm": "fpgrowth",
        "min_sup": 1
    })
    assert res_var["patterns"] == []
    assert "no terms survived" in res_var["note"]

    # 2. Term frequency filtering keeps 0 terms on this fixture
    res_tf = mine_patterns_tool.invoke({
        "category_name": "catB",
        "filtering_method": "term_frequency",
        "algorithm": "fpgrowth",
        "min_sup": 1
    })
    assert res_tf["patterns"] == []
    assert "no terms survived" in res_tf["note"]

    # 3. TF-IDF filtering catA: keeps 'alpha', support 3 (found for min_sup=1, 2, 3; 0 patterns at min_sup=4)
    for min_sup in [1, 2, 3]:
        res_catA = mine_patterns_tool.invoke({
            "category_name": "catA",
            "filtering_method": "tfidf",
            "algorithm": "fpgrowth",
            "min_sup": min_sup
        })
        assert res_catA["patterns"] == [{"pattern": ["alpha"], "support": 3}]

    res_catA_4 = mine_patterns_tool.invoke({
        "category_name": "catA",
        "filtering_method": "tfidf",
        "algorithm": "fpgrowth",
        "min_sup": 4
    })
    assert res_catA_4["patterns"] == []

    # 4. TF-IDF filtering catB: keeps 'gamma', support 4 (found for min_sup=1, 2, 3, 4)
    for min_sup in [1, 2, 3, 4]:
        res_catB = mine_patterns_tool.invoke({
            "category_name": "catB",
            "filtering_method": "tfidf",
            "algorithm": "fpgrowth",
            "min_sup": min_sup
        })
        assert res_catB["patterns"] == [{"pattern": ["gamma"], "support": 4}]


def test_mine_patterns_topk_and_maxfpgrowth_known_answer_20():
    """Test mine_patterns_tool with topk and maxfpgrowth against TEST_FIXTURE.md known answer #20."""
    session = SessionState()

    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    pattern_tools = make_tools(session)
    mine_patterns_tool = next(t for t in pattern_tools if t.name == "mine_patterns_tool")

    # 1. algorithm="topk", testing at k=1, k=2, and k=3 -> returns pattern {'alpha'} support 3 for catA, {'gamma'} support 4 for catB
    for k in [1, 2, 3]:
        res_topk_catA = mine_patterns_tool.invoke({
            "category_name": "catA",
            "filtering_method": "tfidf",
            "algorithm": "topk",
            "k": k
        })
        assert res_topk_catA["patterns"] == [{"pattern": ["alpha"], "support": 3}]

        res_topk_catB = mine_patterns_tool.invoke({
            "category_name": "catB",
            "filtering_method": "tfidf",
            "algorithm": "topk",
            "k": k
        })
        assert res_topk_catB["patterns"] == [{"pattern": ["gamma"], "support": 4}]

    # 2. algorithm="maxfpgrowth", testing specifically for min_sup=1 and min_sup=4 -> returns zero patterns
    for min_sup in [1, 4]:
        res_max_catA = mine_patterns_tool.invoke({
            "category_name": "catA",
            "filtering_method": "tfidf",
            "algorithm": "maxfpgrowth",
            "min_sup": min_sup
        })
        assert res_max_catA["patterns"] == []

        res_max_catB = mine_patterns_tool.invoke({
            "category_name": "catB",
            "filtering_method": "tfidf",
            "algorithm": "maxfpgrowth",
            "min_sup": min_sup
        })
        assert res_max_catB["patterns"] == []


def test_mine_patterns_accented_character():
    """Test mine_patterns_tool when documents contain non-ASCII accented characters."""
    session = SessionState()
    # Create enough repeating terms so term_frequency filtering percentile range keeps them
    session.dataframe = pd.DataFrame({
        "text": ["résumé café tea milk"] * 10,
        "category_name": ["catA"] * 10
    })

    pattern_tools = make_tools(session)
    mine_patterns_tool = next(t for t in pattern_tools if t.name == "mine_patterns_tool")

    for fmt in ["tfidf", "term_frequency"]:
        res = mine_patterns_tool.invoke({
            "category_name": "catA",
            "filtering_method": fmt,
            "algorithm": "fpgrowth",
            "min_sup": 1
        })
        assert "error" not in res
        assert res["total_patterns"] > 0
