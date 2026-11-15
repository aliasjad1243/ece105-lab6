"""The chaos game, as a class.

My Answers to the questions : 

(a) With Rule(), any vertex can come next, so every spot in the square gets reached and it fills in. NotSame() bans picking the same vertex twice in a row, so some spots can never be reached. 
Those empty spots repeat at smaller and smaller sizes, and that pattern of holes is the fractal.

(b) play just calls self.rule.choices(...). Each rule class has its own version of choices, and Python runs the one that matches the rule the game was given.

(c) chaos/__init__.py has the line from .rules import NotSame. Importing chaos runs that fole, so NotSame becomes available straight from chaos."""

import numpy as np
from .rules import Rule


class ChaosGame:
    """One chaos game: the vertices, the fraction f, a rule, and a random
    generator of its own.

    vertices: a (k, 2) array, one vertex per row (or a list of pairs)
    f:        each move goes this fraction of the way toward a vertex
    rule:     a Rule (or subclass): which vertices the next move may go to.
              None means Rule(), any vertex.
    seed:     the same seed plays the same game, move for move
    """

    def __init__(self, vertices, f=0.5, rule=None, seed=None):
        self.vertices = np.asarray(vertices, dtype=float)
        self.f = f
        self.rule = Rule() if rule is None else rule
        self.rng = np.random.default_rng(seed)

    def __repr__(self):
        return (f"ChaosGame({len(self.vertices)} vertices, f={self.f:.3g}, "
                f"rule={self.rule!r})")

    def play(self, n):
        """Play n moves; return the n points as an (n, 2) array.

        Start at the center of the vertices (their mean), with current = 0.
        Then, for each move i = 0, 1, ..., n - 1:
          a. allowed = the rule's choices, for k vertices and current;
          b. current = self.rng.choice(allowed), this move's vertex;
          c. p = p + self.f * (v - p), where v is that vertex (Lecture 13);
          d. row i of the points is p.
        """
        points = np.zeros((n, 2))
        # 4. the start: p, k, and current
        p = self.vertices.mean(axis=0)
        k = len(self.vertices)
        current = 0
        # 5. the n moves, a. to d.
        for i in range(n):
            allowed = self.rule.choices(k, current)   # a.
            current = self.rng.choice(allowed)        # b.
            v = self.vertices[current]
            p = p + self.f * (v - p)                  # c.
            points[i] = p                             # d.
        return points