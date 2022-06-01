from functools import partial
from .base import Sort
# from .trmot import TRMOT as Tracker
# from .stage import TrackerStage as TrackerStage_base

__all__ = ['Tracker', 'TrackerStage', 'Sort']
# TrackerStage_base, Tracker, 
TrackerStage = partial(Sort)
