"""Module to provide primal lattice for color codes."""
from typing import Optional
import numpy as np
import svg
from IPython.display import display, SVG


class PrimalLattice:
    """Primal lattice for color codes."""

    def __init__(self,
                 distance,
                 primal_graph,
                 color_order):
        self.color_order = color_order
        self.primal_graph = primal_graph

        # lattice length
        self.lattice_length = int((distance-1)/2)

        # cols and rows in the grid
        self.face_cols = 2*self.lattice_length
        self.face_rows = 3*self.lattice_length

        # size of hexagon
        self.hexagon_size = 40

        # horizontal separation between hexagons
        self.hor_sep = self.hexagon_size*3/2
        # vertical separation between centers of diagonally stacked hexagons
        self.ver_sep = self.hexagon_size*np.sqrt(3)/2

        # shift grid to keep in svg frame
        self.x_shift = self.hexagon_size*2
        self.y_shift = 3*self.lattice_length*self.ver_sep + self.hexagon_size/2

        # color map
        self.color_map = {'r': '#FA8072', 'g': '#33FF93', 'b': '#069AF3'}

        # face coordinates
        for f in primal_graph.faces:
            x = self.hexagon_size + self.hor_sep*f[0]
            y = self.ver_sep*f[1]
            primal_graph.faces[f]['pos_graph'] = (x, y)
            primal_graph.faces[f]['pos_lat'] = \
                (self.x_shift + x,
                 self.y_shift - y)

        # vertex coordinates
        def xe(i):
            return self.hexagon_size * (-1)**i * (-1 + (-1)**i * (1 + 6*i))/4
        for node in primal_graph.nodes:
            x = xe(node[0])
            if node[1] % 2 == 1:
                x = x + self.hexagon_size/2 if node[0] % 2 == 0 \
                    else x - self.hexagon_size/2
            y = node[1] * self.ver_sep
            primal_graph.nodes[node]['pos_graph'] = (x, y)
            primal_graph.nodes[node]['pos_lat'] = \
                (self.x_shift + x, self.y_shift - y)

    def _create_face_svg(self,
                         face):
        """
        Create an svg.Polygon object for a face.

        Parameters
        ----------
        x0 : float
            Horizontal position of center.
        y0 : float
            Vertical position of center.
        color : str
            Options are r, g or b.

        Returns
        -------
        pg : svg.Polygon
            The svg.Polygon object of the hexagon.

        """
        coords = [self.primal_graph.nodes[node]['pos_lat']
                  for node in self.primal_graph.faces[face]['nodes']]

        pts = [a for p in coords for a in p]

        c = self.color_order[self.primal_graph.faces[face]['color']]

        pg = svg.Polygon(
            points=pts,
            fill=self.color_map[c],
            stroke="black",
            stroke_width=1,
            stroke_linejoin="round",
        )
        return pg

    def setup_draw(self,
                   draw_boundaries: bool = False,
                   draw_vertex_labels: Optional[int] = None,
                   draw_face_labels: Optional[int] = None
                   ) -> None:
        """
        Set the options for drawing the primal lattice.

        The `draw` function can be used to display the lattice.

        Parameters
        ----------
        draw_boundaries : bool, optional
            Draw the boundaries of the lattice. The default is False.
        draw_vertex_labels : Optional[int], optional
            Draw the vertex labels. The default is None.
        draw_face_labels : Optional[int], optional
            Draw the face labels. The default is None.
        """
        self._svg_els = []

        # draw the boundaries first
        if draw_boundaries:
            right_corner_x = self.face_cols * self.hor_sep + self.x_shift
            pts = [self.x_shift, self.y_shift+5,
                   self.x_shift, self.y_shift,
                   right_corner_x, self.y_shift,
                   right_corner_x, self.y_shift+5]
            # bottom boundary
            self._svg_els.append(
                svg.Polygon(
                    points=pts,
                    fill='blue',
                    stroke="black",
                    stroke_width=1,
                    stroke_linejoin="round",

                ))
            # left boundary
            self._svg_els.append(
                svg.Polygon(
                    points=pts,
                    fill='red',
                    stroke="black",
                    stroke_width=1,
                    stroke_linejoin="round",
                    transform=f'rotate(-60 {self.x_shift-5} {self.y_shift})'
                ))
            # right boundary
            self._svg_els.append(
                svg.Polygon(
                    points=pts,
                    fill='green',
                    stroke="black",
                    stroke_width=1,
                    stroke_linejoin="round",
                    transform=f'rotate(60 {right_corner_x+5} {self.y_shift})'
                ))
        for face in self.primal_graph.faces:
            self._svg_els.append(
                self._create_face_svg(face)
            )

        if type(draw_vertex_labels) is int:
            face_color = [0, 1, 2] if draw_vertex_labels == 3 \
                else [draw_vertex_labels]
            included_faces = [f for f in self.primal_graph.faces
                              if self.primal_graph.faces[f]['color']
                              in face_color]
            for face in included_faces:
                for node in self.primal_graph.faces[face]['nodes']:
                    node_pos = self.primal_graph.nodes[node]['pos_lat']
                    self._svg_els.append(
                        svg.Text(x=node_pos[0], y=node_pos[1],
                                 text=f'{node[0]},{node[1]}',
                                 font_size=8,
                                 text_anchor='middle'))

        if type(draw_face_labels) is int:
            face_color = [0, 1, 2] if draw_face_labels == 3 \
                else [draw_face_labels]
            included_faces = [f for f in self.primal_graph.faces
                              if self.primal_graph.faces[f]['color']
                              in face_color]
            for face in included_faces:
                face_pos = self.primal_graph.faces[face]['pos_lat']
                self._svg_els.append(svg.Text(x=face_pos[0], y=face_pos[1],
                                              text=f'{face[0]},{face[1]}',
                                              font_size=10,
                                              text_anchor='middle'))

    def label_vertex(self,
                     label: str,
                     node: tuple) -> None:
        """
        Label a vertex on the lattice to be drawn.

        Parameters
        ----------
        label : str
            Label to include. One character for nice display.
        node : tuple
            The address of the node at which to place the label..
        """
        node_pos = self.primal_graph.nodes[node]['pos_lat']
        self._svg_els.append(
            svg.Circle(cx=node_pos[0], cy=node_pos[1],
                       r=8,
                       fill='white',
                       stroke=None))
        self._svg_els.append(
            svg.Text(x=node_pos[0], y=node_pos[1]+2.5,
                     text=label,
                     font_size=10,
                     text_anchor='middle'))

    def label_operator(self,
                       operator: np.ndarray
                       ) -> None:
        """
        Label an operator on the lattice to be drawn.

        Parameters
        ----------
        operator : np.ndarray
            A one-dimensional numpy array of the operator. with length twice
            the number of qubits in the code. Entries should be 0 or 1.
        """

        n = int(len(operator)/2)
        for i in range(n):
            if operator[i] and not operator[n+i]:
                self.label_vertex('X', self.primal_graph.nodes_index.inv[i])
            elif operator[i] and operator[n+i]:
                self.label_vertex('Y', self.primal_graph.nodes_index.inv[i])
            elif not operator[i] and operator[n+i]:
                self.label_vertex('Z', self.primal_graph.nodes_index.inv[i])

    def label_face(self,
                   label: str,
                   face: tuple
                   ) -> None:
        """
        Label a face on the lattice to be drawn.

        Parameters
        ----------
        label : str
            The string to be placed on the face.
        face : tuple
            The address of the face which is to be labelled.
        """
        node_pos = self.primal_graph.faces[face]['pos_lat']
        self._svg_els.append(
            svg.Rect(x=node_pos[0]-5, y=node_pos[1]-5,
                     width=10, height=10,
                     fill='white',
                     stroke=None))
        self._svg_els.append(
            svg.Text(x=node_pos[0], y=node_pos[1]+2.5,
                     text=label,
                     font_size=10,
                     text_anchor='middle'))

    def label_syndrome(self,
                       syndrome: np.ndarray
                       ) -> None:
        """
        Label a syndrome on the lattice to be drawn.

        Parameters
        ----------
        syndrome : np.ndarray
            A one-dimensional numpy array. Length should be equal to the number
            of generators of the code. Entries should be 0 or 1.
        """
        m = int(len(syndrome)/2)
        for i in range(m):
            if syndrome[i] and not syndrome[m+i]:
                self.label_face('X', self.primal_graph.faces_index.inv[i])
            elif syndrome[i] and syndrome[m+i]:
                self.label_face('Y', self.primal_graph.faces_index.inv[i])
            elif not syndrome[i] and syndrome[m+i]:
                self.label_face('Z', self.primal_graph.faces_index.inv[i])

    def draw(self
             ) -> None:
        """Display the primal lattice with any labels put on it."""
        img = svg.SVG(
            width=self.face_cols*self.hor_sep + 3*self.hexagon_size,
            height=self.face_rows*self.ver_sep + 1*self.hexagon_size,
            elements=self._svg_els)
        display(SVG(img.as_str()))
