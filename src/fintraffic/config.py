"""Experimental settings used in the paper."""

#: Sliding-window sizes.
WINDOWS = list(range(5, 71, 5))

#: Every window forecasts from the same index, so all of them are evaluated on
#: exactly the same targets (otherwise window 5 would be scored on 65 more points).
TEST_START = max(WINDOWS)

#: q grid of the exploratory run (original 0.1..1.0 grid extended beyond q = 1).
Q_GRID = [round(i / 10, 1) for i in range(1, 11)] + [1.25, 1.5, 1.75, 2.0, 2.5, 3.0]

#: q grid of the tables reported in the paper (0.1 to 10, denser around q = 3).
Q_PAPER = [0.1, 0.5, 1.0, 2.0, 2.5, 3.0, 3.5, 4.0, 6.0, 10.0]

#: Dense q grid for the CC-min "validation MAE vs q" curves.
Q_CURVE = [0.1, 0.2, 0.5, 0.75, 1, 1.25, 1.5, 1.75,
           2, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9,
           3, 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8, 3.9,
           4, 5, 6, 8, 10]

#: Windows shown in the CC-min q-curve figure.
CURVE_WINDOWS = [5, 10, 15, 20, 25, 30, 35, 40]
