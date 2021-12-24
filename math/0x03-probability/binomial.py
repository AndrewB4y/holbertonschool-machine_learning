#!/usr/bin/env python3


"""
Binomial module
"""


class Binomial:
    """
    Binomial distribution class
    """
    e = 2.7182818285
    pi = 3.1415926536

    def __init__(self, data=None, n=1, p=0.5):
        """
        Binomial class initializer
        @data: is a list of the data to be used to estimate the distribution.
        @n: is the number of Bernoulli trials.
        @p: is the probability of a “success”
        """

        if data is None:
            if n <= 0:
                raise ValueError("n must be a positive value")
            if p <= 0 or p >= 1:
                raise ValueError(
                    "p must be greater than 0 and less than 1"
                )
            self.n = int(n)
            self.p = float(p)
            return

        if type(data) is not list:
            raise TypeError("data must be a list")
        elif len(data) <= 1:
            raise ValueError("data must contain multiple values")

        self.mean = float(sum(data)/len(data))
        self.variance = sum(
            [(x - self.mean)**2 for x in data]
        ) / float(len(data))
        self.stddev = self.variance**(0.5)

        self.p = 1 - (self.variance / self.mean)
        self.n = round(self.mean / self.p)
        self.p = float(self.mean / self.n)

    def factorial(self, n):
        """
        factorial(self, n) - calculates the factorial of n
        """
        if n <= 0:
            return 1

        res = 1
        for i in range(1, n + 1):
            res = res * i
        return res

    def pmf(self, k):
        """
        pmf(self, k) - Calculates the value of the PMF for a given
                       number of “successes”.
        @k: is the number of “successes".
        """

        k = int(k)
        if k < 0 or k > self.n:
            return 0

        n_k = self.factorial(self.n)
        n_k = n_k / (self.factorial(k) * self.factorial(self.n - k))
        f = n_k * (self.p**k) * ((1 - self.p)**(self.n - k))

        return f

    def cdf(self, k):
        """
        cdf(self, k) - Calculates the value of the cumulative distribution
                       function (CDF) for a number of “successes”.

        @k: is the number of “successes"
        Returns: the CDF value for @k
        """

        k = int(k)
        if k < 0 or k > self.n:
            return 0
        cdf = 0.0
        for value in range(0, k + 1):
            cdf += self.pmf(value)

        return cdf