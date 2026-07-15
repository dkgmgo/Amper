"""Tests for graph loading utilities."""

import math

import networkx as nx
import pytest

from datasets import load_graph, load_record, parse_country_snapshot


def test_parse_country_snapshot(tmp_path):
    p = tmp_path / "FR_1702188000.1500.graphml"
    assert parse_country_snapshot(p) == ("FR", 1702188000)
    p2 = tmp_path / "whatever.graphml"
    assert parse_country_snapshot(p2) == ("whatever", None)


def test_load_graph_custom_feature_attr(tmp_path):
    G = nx.Graph()
    G.add_edge("a", "b", ricci=0.5)
    G.add_edge("b", "c", ricci=1.5)
    path = tmp_path / "XX_1700000000.graphml"
    nx.write_graphml(G, path)
    H = load_graph(path, feature_attr="ricci")
    assert H.number_of_edges() == 2


def test_load_graph_rejects_all_invalid(tmp_path):
    G = nx.Graph()
    G.add_edge("a", "b", distance="not-a-number")
    path = tmp_path / "XX_1700000000.graphml"
    nx.write_graphml(G, path)
    with pytest.raises(ValueError):
        load_graph(path)


def test_load_record_label(tmp_path):
    G = nx.Graph()
    G.add_edge("a", "b", distance=1.0)
    path = tmp_path / "DE_1702188000.graphml"
    nx.write_graphml(G, path)
    rec = load_record(path)
    assert rec.country == "DE"
    assert rec.label.startswith("DE_2023-12-10")
