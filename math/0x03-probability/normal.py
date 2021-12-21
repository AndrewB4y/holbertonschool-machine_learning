#!/usr/bin/env python3


"""
Normal module
"""


class Normal:
    """
    Normal distribution class
    """
    e = 2.7182818285
    pi = 3.1415926536

    def __init__(self, data=None, mean=0., stddev=1.):
        """
        Normal class initializer
        @data: is a list of the data to be used to estimate the distribution.
        @mean: is the mean of the distribution.
        @stddev: is the standard deviation of the distribution.
        """

        if data is None:
            if stddev <= 0:
                raise ValueError("stddev must be a positive value")
            self.stddev = float(stddev)
            self.mean = float(mean)
            return

        if type(data) is not list:
            raise TypeError("data must be a list")
        elif len(data) <= 1:
            raise ValueError("data must contain multiple values")

        self.mean = float(sum(data)/len(data))
        self.stddev = sum(
            [(x - self.mean)**2 for x in data]
            ) / float(len(data))
        self.stddev = self.stddev**(0.5)

    def z_score(self, x):
        """
        z_score(self, x) - Calculates the z-score of a given x-value

        @x is the x-value
        """
        return (x - self.mean) / self.stddev

    def x_value(self, z):
        """
        x_value(self, z) - Calculates the x-value of a given z-score
        """
        return (z * self.stddev) + self.mean

    def pdf(self, x):
        """
        pdf(self, x) - Probability density function method
        """

        if x < 0:
            return 0

        p = (x - self.mean)**2.0 / self.stddev**2.0
        p = Normal.e**(p / -2.0)
        p = p / (self.stddev * ((Normal.pi * 2.0)**0.5))

        return p

    def err_func(self, x):
        """
        err_func(self, x) - calculates de error function for given x.
        """

        err = 2 / (Normal.pi**0.5)
        err = err * (
            x - (x**3) / 3 + (x**5) / 10 - (x**7) / 42 + (x**9) / 216
        )
        return err

    def cdf(self, x):
        """
        cdf(self, x) - Calculates the value of the cumulative distribution
                       function (CDF) for a given time period.

        @x: is the number of “successes"
        Returns: the CDF value for @x
        """

        if x < 0:
            return 0

        cdf = (
            1 + self.err_func((x - self.mean) / (self.stddev * (2**0.5)))
        ) / 2

        return cdf
