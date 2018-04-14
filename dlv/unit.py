class Unit:
    def __init__(self, isFilter, unit):
        self._isFilter = isFilter
        self._unit = unit
        self._activation = []
        self._selectivity = 0.0
