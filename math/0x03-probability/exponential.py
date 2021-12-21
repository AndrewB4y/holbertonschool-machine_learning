#!/usr/bin/env python3


"""
Exponential module
"""


class Exponential:
    """
    Exponential class
    """
    e = 2.7182818285

    def __init__(self, data=None, lambtha=1.):
        """
        Exponential class initializer
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

        self.lambtha = len(data)/sum(data)

    def pdf(self, x):
        """
        pdf(self, x) - Probability density function method
        """

        if x < 0:
            return 0
        p = self.lambtha * ((Exponential.e)**(-self.lambtha * x))
        return p

    def cdf(self, x):
        """
        cdf(self, x) - Calculates the value of the cumulative distribution
                       function (CDF) for a given time period.

        @x: is the number of “successes"
        Returns: the CDF value for @x
        """
        x = int(x)
        if x < 0:
            return 0
        cdf = 1 - (Exponential.e**(-self.lambtha * x))
        return cdf
