#!/usr/bin/env python3

import numpy as np
Binomial = __import__('binomial').Binomial

np.random.seed(0)
data = np.random.binomial(50, 0.6, 100).tolist()
b1 = Binomial(data)
print('P(30):', b1.pmf(30))

b2 = Binomial(n=50, p=0.6)
print('P(30):', b2.pmf(30))


np.random.seed(7)
p = np.random.uniform(0.01, 0.99)
n = np.random.randint(1, 100)
s = np.random.randint(100, 1000)
data = np.random.binomial(n, p, s).tolist()
b = Binomial(data)
k = np.random.randint(1, n)
print(np.format_float_scientific(b.pmf(k), precision=10))
k = np.random.uniform(1.0, n)
print(np.format_float_scientific(b.pmf(k), precision=10))
print(np.around(b.pmf(0), 10))
