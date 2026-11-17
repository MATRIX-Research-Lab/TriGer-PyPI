"""TriGer: structured cross-omics interaction discovery with a triple-graph model."""
from .greedy_peeling import (greedy_peeling_XY_one, greedy_peeling_XY_all,
                             greedy_peeling_X_one, greedy_peeling_X_all,
                             sort_result, sort_result_XandY, greedy_3step)
from .plotting import plot3in1_v1, plot_all_result, matlab_jet

__version__ = "0.1.0"
__all__ = ["greedy_peeling_XY_one", "greedy_peeling_XY_all", "greedy_peeling_X_one",
           "greedy_peeling_X_all", "sort_result", "sort_result_XandY", "greedy_3step",
           "plot3in1_v1", "plot_all_result", "matlab_jet"]
