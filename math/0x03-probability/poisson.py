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

    def pmf(self, k):
        """
        pmf(self, k) - Probability mass function method
        """

        k = int(k)
        if k < 0:
            return 0
        p = ((Poisson.e)**(-self.lambtha)) * (self.lambtha**k)
        k_fact = 1
        for i in range(1, k + 1):
            k_fact = k_fact * i
        p = p / k_fact
        return p

    def cdf(self, k):
        """
        cdf(self, k) - Calculates the value of the cumulative distribution
                       function (CDF) for a given number of “successes”.

        @k: is the number of “successes"
        Returns: the CDF value for @k
        """
        k = int(k)
        if k < 0:
            return 0
        cdf = 0.0
        for value in range(0, k + 1):
            cdf += self.pmf(value)

        return cdf
