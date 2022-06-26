import itertools
from math import factorial

from attr import attr

from constants.constants import LOWER_BBAND_STD1, LOWER_BBAND_STD2, UPPER_BBAND_STD1, UPPER_BBAND_STD2

class ComboGenerator:
    
    def __init__(self, persistent_attributes, attribute_pool, r=3):
        self.persistent_attributes = persistent_attributes
        self.combos = itertools.combinations(attribute_pool, r)

        n = len(attribute_pool)
        self.num_combos = int(factorial(n) / (factorial(r) * factorial(n - r)))


    def get_next_combo(self):
        combo_attributes = list(next(self.combos))
        combo =  self.persistent_attributes + combo_attributes

        flat_combo = []
        for attr_group in combo:
            if isinstance(attr_group, list):
                for attr in attr_group:
                    flat_combo.append(attr)
            else:
                flat_combo.append(attr_group)
        return flat_combo


    def get_num_combos(self):
        return self.num_combos


    def combo_is_valid(self, combo):

        if UPPER_BBAND_STD1 in combo and not LOWER_BBAND_STD1 in combo:
            return False
        elif UPPER_BBAND_STD2 in combo and not LOWER_BBAND_STD2 in combo:
            return False

        return True