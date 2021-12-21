#!/usr/bin/env python3


"""
poisson module
"""


class Poisson:
    """
    Poisson class
    """
    e = 2.7182818285

    def __init__(self, data=None, lambtha=1.):
        """
        Poisson class initializer
        @data: is a list of the data to be used to estimate the distribution.
        @lambtha: is the expected number of occurences in a given time frame.
        """

        if data is None:
            if lambtha <= 0:
                raise ValueError("lambtha must be a positive value")
            self.lambtha = float(lambtha)
            return

        if type(data) is not list:
            raise TypeError("data must be a list")
        elif len(data) <= 1:
            raise ValueError("data must contain multiple values")

        self.lambtha = sum(data)/len(data)
