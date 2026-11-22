"""TriGer: structured cross-omics interaction discovery with a triple-graph model."""
from .greedy_peeling import (Block, Community, greedy_peeling_XY_one,
                             greedy_peeling_XY_all, greedy_peeling_X_one,
                             greedy_peeling_X_all, sort_result,
                             sort_result_XandY, greedy_3step)
from .association import (Association, Module, TrigerResult,
                          association_matrices, triger)
from .plotting import plot3in1_v1, plot_all_result, matlab_jet

__version__ = "1.0.0"
__all__ = ["triger", "association_matrices", "greedy_3step",
           "greedy_peeling_XY_one", "greedy_peeling_XY_all",
           "greedy_peeling_X_one", "greedy_peeling_X_all", "sort_result",
           "sort_result_XandY", "plot3in1_v1", "plot_all_result",
           "matlab_jet", "Block", "Community", "Module", "Association",
           "TrigerResult"]
