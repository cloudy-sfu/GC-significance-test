from pyvis.network import Network
import networkx as nx
import numpy as np
from bs4 import BeautifulSoup


def draw(adjacency: np.ndarray, labels: list, output_path: str):
    g = nx.DiGraph()
    g.add_edges_from(zip(*np.where(adjacency)))
    g = nx.relabel_nodes(g, {i: labels[i] for i in range(len(labels))})
    h = Network('100vh', '100%', directed=True)
    h.from_nx(g)
    h = BeautifulSoup(h.generate_html(), 'html.parser')
    [s.extract() for s in h.find('head').find_all('center')]
    s = h.find('body').find('div', {'class': 'card'})
    s.replaceWith(s.find('div', {'id': 'mynetwork'}))
    with open(output_path, 'w') as f:
        f.write(str(h.prettify()))
