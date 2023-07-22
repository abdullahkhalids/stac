"""Module to provide color code."""
from typing import Any, Optional
from functools import partial
import numpy as np
import networkx as nx
import bidict
from ordered_set import OrderedSet
import matplotlib.pyplot as plt

from stac import Circuit
from ..code import Code
from .primallattice import PrimalLattice


class ColorCode(Code):
    """Class for creating triangular color codes."""

    def __init__(self,
                 distance: int,
                 geometry: str = "hexagonal",
                 color_order: list[str] = ['g', 'r', 'b']
                 ) -> None:
        """
        Construct the color code of some geometry and distance.

        Parameters
        ----------
        distance : int
            The distance of the code.
        geometry : str, optional
            Describes the shape of the primal lattice. The default and only
            option currently is "hexagonal".
        color_order: str, optional
            Order of colors in the lattice.
        """
        self.color_order = color_order
        # lattice length
        L = int((distance-1)/2)

        # rows of hexagons
        rows = 3*L

        # primal graph
        self.primal_graph = nx.Graph()
        self.primal_graph.faces = dict()
        self.primal_graph.boundaries = dict()
        self.primal_graph.draw = self._primal_graph_draw

        # determine nodes and edges of primal graph
        self.primal_graph.nodes_index = bidict.bidict()
        kk = 0
        heights = [i for i in range(2, rows, 3)] + \
            [i for i in range(rows + 1, 0, -3)]
        row_nodes = [[] for i in range(rows+1)]
        for i in range(distance):
            for j in range(heights[i]):
                self.primal_graph.add_node((i, j),
                                           faces=OrderedSet())
                self.primal_graph.nodes_index[(i, j)] = kk
                kk += 1
                row_nodes[j].append((i, j))
                if j != 0:
                    self.primal_graph.add_edge((i, j), (i, j-1))

        b = 0
        for row in row_nodes:
            for i in range(len(row)-1):
                if b:
                    self.primal_graph.add_edge(row[i], row[i+1])
                b = (b+1) % 2
            b = (b+1) % 2

        # boundaries
        self.primal_graph.boundaries[2] = \
            OrderedSet([(i, 0) for i in range(distance)])
        self.primal_graph.boundaries[1] = \
            OrderedSet([row[0] for i, row in enumerate(row_nodes)
                        if i % 3 in [0, 1]])
        self.primal_graph.boundaries[0] = \
            OrderedSet([row[-1] for i, row in enumerate(row_nodes)
                        if i % 3 in [0, 2]])

        # faces
        f_heights = [i+1 for i in range(2, 3*L, 3)] + \
            [i+1 for i in range(3*L-2, 0, -3)]
        self.primal_graph.faces_index = bidict.bidict()
        kk = 0

        c_bot = 0
        c = 0
        for i in range(2*L):
            for j in range(c_bot, f_heights[i], 2):
                self.primal_graph.faces[(i, j)] = dict()
                if j == 0 and c == 0:
                    self.primal_graph.faces[(i, j)]['form'] = 'bottom'
                elif i < L and j == f_heights[i]-1:
                    self.primal_graph.faces[(i, j)]['form'] = 'right'
                elif i >= L and j == f_heights[i]-1:
                    self.primal_graph.faces[(i, j)]['form'] = 'left'
                else:
                    self.primal_graph.faces[(i, j)]['form'] = 'full'

                nodes = OrderedSet()

                for di, dj in [(1, 1), (1, 0), (1, -1),
                               (0, -1), (0, 0), (0, 1)]:
                    if j+dj < 0 or j+dj >= heights[i+di]:
                        continue
                    nodes.add((i+di, j+dj))

                self.primal_graph.faces[(i, j)]['nodes'] = nodes
                for v in nodes:
                    self.primal_graph.nodes[v]['faces'].add((i, j))

                self.primal_graph.faces[(i, j)]['color'] = c
                c = (c+2) % 3

                self.primal_graph.faces_index[(i, j)] = kk
                kk += 1

            c_bot = (c_bot+1) % 2
            c = c_bot

        # add color to edges
        all_boundary_nodes = self.primal_graph.boundaries[0] | \
            self.primal_graph.boundaries[1] | \
            self.primal_graph.boundaries[2]

        # add color to all edges that terminate on boundaries
        for j in range(3):
            for v in self.primal_graph.boundaries[j]:
                for e in self.primal_graph.edges(v):
                    if e[1] not in all_boundary_nodes:
                        self.primal_graph.edges[e]['color'] = j
        # add color to edges from corner
        self.primal_graph.edges[(L, rows), (L, rows-1)]['color'] = 1
        self.primal_graph.edges[(0, 0), (0, 1)]['color'] = 2
        self.primal_graph.edges[(2*L, 0), (2*L-1, 0)]['color'] = 0

        # add color to edges in bulk
        for e in self.primal_graph.edges:
            if 'color' in self.primal_graph.edges[e]:
                continue
            face = self.primal_graph.nodes[e[0]]['faces'].difference(
                self.primal_graph.nodes[e[1]]['faces'])[0]
            self.primal_graph.edges[e]['color'] = \
                self.primal_graph.faces[face]['color']

        # corner colors
        self.primal_graph.corners = dict()
        self.primal_graph.corners[0] = (0, 0)
        self.primal_graph.corners[1] = (2*L, 0)
        self.primal_graph.corners[2] = (L, 3*L)

        self.primal_lattice = PrimalLattice(distance,
                                            self.primal_graph,
                                            color_order)

        # create the generator matrix
        n = int(3*(distance-1)*(distance+1)/4+1)
        mhalf = int((n-1)/2)
        H = np.zeros((mhalf, n), dtype=int)

        for i, f in self.primal_graph.faces_index.inv.items():
            for v in self.primal_graph.faces[f]['nodes']:
                v_ind = self.primal_graph.nodes_index[v]
                H[i, v_ind] = 1

        generator_matrix = np.zeros((2*mhalf, 2*n), dtype=int)
        generator_matrix[:mhalf, :n] = H
        generator_matrix[mhalf:, n:] = H

        super().__init__(generator_matrix)
        self.num_generators_x = mhalf
        self.num_generators_z = mhalf
        self.distance = distance

    def construct_logical_operators(self,
                                    method: str = "boundary: blue"
                                    ) -> (Any, Any):
        """
        Construct logical operators of the code.

        Parameters
        ----------
        method : str, optional
            With boundaries with color 0, 1, 2. The options are:
                "boundary: green"
                "boundary: red"
                "boundary: blue" (default)
                "gottesman" (generic method)

        Returns
        -------
        logical_xs: numpy.array
            Array of logical xs. Each row is an operator.
        logical_zs: numpy.array
            Array of logical xs. Each row is an operator.
        """
        if method == "boundary: green":
            c = self.primal_lattice.color_order.index('g')
        elif method == "boundary: red":
            c = self.primal_lattice.color_order.index('r')
        elif method == "boundary: blue":
            c = self.primal_lattice.color_order.index('b')
        else:
            return super().construct_logical_operators(method)

        oper_x = np.zeros(2*self.num_data_qubits, dtype=int)
        oper_z = np.zeros(2*self.num_data_qubits, dtype=int)
        for node in self.primal_graph.boundaries[c]:
            oper_x[self.primal_graph.nodes_index[node]] = 1
            oper_z[self.num_data_qubits +
                   self.primal_graph.nodes_index[node]] = 1

        self.logical_xs = np.array([oper_x])
        self.logical_zs = np.array([oper_z])

        return self.logical_xs, self.logical_zs

    def _primal_graph_draw(self,
                           draw_face_labels: bool = True
                           ) -> None:
        """
        Draw the primal graph.

        Parameters
        ----------
        draw_face_labels : bool, optional
            Draw the face labels. The default is True.
        """
        # nicer labels
        if not hasattr(self.primal_graph, '_node_labels'):
            self.primal_graph._node_labels = dict()
            for node in self.primal_graph.nodes:
                self.primal_graph._node_labels[node] = str(node)[1:-1]

        nx.draw(self.primal_graph,
                pos=nx.get_node_attributes(self.primal_graph, 'pos_graph'),
                node_size=450,
                font_size=7,
                labels=self.primal_graph._node_labels,
                with_labels=True)

        if draw_face_labels:
            # nicer labels
            if not hasattr(self.primal_graph, '_face_labels'):
                self.primal_graph._face_labels = dict()
                for face in self.primal_graph.faces:
                    self.primal_graph._face_labels[face] = str(face)[1:-1]

            plt.axis('off')
            pos = {f: val['pos_graph']
                   for f, val in self.primal_graph.faces.items()}

            nx.draw_networkx_labels(self.primal_graph,
                                    pos=pos,
                                    labels=self.primal_graph._face_labels,
                                    font_size=7)

    def construct_dual_graph(self):
        """
        Construct the dual graph of the code.

        In the dual graph, the stabilizers are mapped onto the vertices and
        the qubits are mapped onto the faces. The stabilizers refer to both
        the set of pure X stabilizers of the code, and the pure Z ones. The
        vertices are colored, like the faces of the primal lattice.
        """
        self.dual_graph = nx.Graph()
        # nodes of dual graph are faces of primal
        for face, val in self.primal_graph.faces.items():
            pos = (-self.primal_lattice.x_shift + val['pos_lat'][0],
                   self.primal_lattice.y_shift - val['pos_lat'][1])
            self.dual_graph.add_node(face,
                                     color=val['color'],
                                     pos_graph=pos,
                                     faces=val['nodes'])
        self.dual_graph.nodes_index = self.primal_graph.faces_index

        # now add edges between the new nodes
        for face in self.primal_graph.faces:
            for d in [(-1, -1), (-1, 1), (0, -2), (0, 2), (1, -1), (1, 1)]:
                connected_face = (face[0] + d[0], face[1] + d[1])
                if connected_face in self.primal_graph.faces:
                    self.dual_graph.add_edge(face, connected_face)

        # color of nodes has to be a list for draw function
        self.dual_graph.node_colors = \
            [self.primal_lattice.color_map[self.primal_lattice.color_order[c]]
             for f, c in nx.get_node_attributes(
                self.dual_graph, 'color').items()]

        # now add faces, which are triangles
        self.dual_graph.faces = dict()
        self.dual_graph._face_labels = dict()
        for node in self.primal_graph.nodes:
            self.dual_graph.faces[node] = dict()
            self.dual_graph.faces[node]['nodes'] = \
                self.primal_graph.nodes[node]['faces']
            self.dual_graph.faces[node]['pos_graph'] = \
                self.primal_graph.nodes[node]['pos_graph']
            self.dual_graph._face_labels[node] = str(node)[1:-1]
        self.dual_graph.faces_index = self.primal_graph.nodes_index

        # edges index
        self.dual_graph.edges_index = bidict.bidict()
        kk = 0
        for e in self.dual_graph.edges:
            self.dual_graph.edges_index[frozenset(e)] = kk
            kk += 1

        # nicer labels
        self.dual_graph._node_labels = dict()
        for node in self.dual_graph.nodes:
            self.dual_graph._node_labels[node] = str(node)[1:-1]
        # draw
        self.dual_graph.draw = self._dual_graph_draw

    def _dual_graph_draw(self,
                         draw_vertex_labels: bool = True,
                         draw_face_labels: bool = True,
                         edge_list: Optional[list] = None,
                         highlight_nodes: Optional[list] = None
                         ) -> None:
        """
        Draw the dual graph.

        Parameters
        ----------
        draw_vertex_labels : bool, optional
            Draw the vertex labels. The default is True.
        draw_face_labels : bool, optional
            Draw the face labels. The default is False.
        edge_list: list, optional
            List of edges to draw
        highlight_nodes: list, optional
            List of nodes to highlight
        """
        plt.figure(figsize=(10, 8))
        plt.axis('off')

        pos = nx.get_node_attributes(self.dual_graph, 'pos_graph')
        if highlight_nodes:
            nx.draw_networkx_nodes(self.dual_graph,
                                   pos=pos,
                                   nodelist=highlight_nodes,
                                   node_size=450,
                                   node_shape='s',
                                   node_color='orange')

        if not edge_list:
            edge_list = list(self.dual_graph.edges())

        nx.draw(self.dual_graph,
                pos=pos,
                node_color=self.dual_graph.node_colors,
                node_size=450,
                font_size=7,
                labels=self.dual_graph._node_labels,
                with_labels=True,
                edgelist=edge_list)

        if draw_face_labels:
            plt.axis('off')
            pos = {f: val['pos_graph']
                   for f, val in self.dual_graph.faces.items()}
            nx.draw_networkx_nodes(self.primal_graph,
                                   pos=pos,
                                   node_color='white')
            nx.draw_networkx_labels(self.primal_graph,
                                    pos=pos,
                                    labels=self.dual_graph._face_labels,
                                    font_size=7)

    def construct_restricted_graphs(self):
        """
        Construct the restricted graphs.

        There are three restricted graphs. Each is built by omitting vertices
        of one color from the dual graph.

        The graphs are stored in the dictionary `self.restricted_graphs`. There
        are three keys for this dictionary, (0, 1), (0, 2), and (1, 2),
        referring to the colors that are included in the graph.
        """
        self.restricted_graphs = dict()
        for c1, c2 in [(0, 1), (0, 2), (1, 2)]:
            self.restricted_graphs[c1, c2] = nx.Graph()

            # add nodes
            for node, val in self.dual_graph.nodes.items():
                if val['color'] in [c1, c2]:
                    self.restricted_graphs[c1, c2].add_node(node, **val)

            # add edges
            for node in self.dual_graph.nodes:
                if self.dual_graph.nodes[node]['color'] != c1:
                    continue
                for e in nx.edges(self.dual_graph, node):
                    if self.dual_graph.nodes[e[1]]['color'] == c2:
                        self.restricted_graphs[c1, c2].add_edge(*e)
                        self.restricted_graphs[c1, c2].edges[e]['faces'] = \
                            self.primal_graph.faces[e[0]]['nodes'] & \
                            self.primal_graph.faces[e[1]]['nodes']

            self.restricted_graphs[c1, c2].node_colors = \
                [self.primal_lattice.color_map[
                    self.primal_lattice.color_order[c]]
                 for f, c in nx.get_node_attributes(
                    self.restricted_graphs[c1, c2], 'color').items()]

            self.restricted_graphs[c1, c2].draw = partial(
                self._restricted_graph_draw, (c1, c2))
            self.restricted_graphs[c1, c2].draw.__doc__ = \
                """\
                Draw this restricted graph.

                Parameters
                ----------
                draw_edge_labels : bool, optional
                    Draw the edge labels. The default is False.
                """

    def _restricted_graph_draw(self,
                               label: tuple,
                               draw_edge_labels: bool = False):
        """
        Draw a restricted graph.

        Parameters
        ----------
        label : tuple
            The label of the restricted graph: (0, 1), (0, 2) or (1, 2).
        draw_edge_labels : bool, optional
            Draw the edge labels. The default is False.
        """
        node_labels = dict()
        for node in self.restricted_graphs[label].nodes:
            node_labels[node] = str(node)[1:-1]

        pos = nx.get_node_attributes(
            self.restricted_graphs[label], 'pos_graph')

        nx.draw(self.restricted_graphs[label],
                pos=pos,
                node_color=self.restricted_graphs[label].node_colors,
                node_size=450,
                font_size=7,
                labels=node_labels,
                with_labels=True)

        if draw_edge_labels:
            edge_labels = dict()

            for e, val in self.restricted_graphs[label].edges.items():
                fs = list(val['faces'])
                edge_labels[e] = str(fs[0])[1:-1] + '\n' + str(fs[1])[1:-1]

            nx.draw_networkx_edge_labels(self.restricted_graphs[label],
                                         pos=pos,
                                         edge_labels=edge_labels,
                                         font_size=7,
                                         verticalalignment='center_baseline')
