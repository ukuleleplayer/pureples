"""
The substrate.
"""


class Substrate(object):
    """
    Represents a substrate: Input coordinates, output coordinates, hidden coordinates and a resolution defaulting to 10.0.
    """

    def __init__(self, input_coordinates, output_coordinates, hidden_coordinates=(), res=10.0):
        self.input_coordinates = input_coordinates
        self.hidden_coordinates = hidden_coordinates
        self.output_coordinates = output_coordinates
        self.res = res
