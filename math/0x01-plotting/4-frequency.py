#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(5)
student_grades = np.random.normal(68, 15, 50)

bins = np.arange(0, 100, 10)
n, bins, patches = plt.hist(student_grades, bins=bins, edgecolor='k')


plt.xlabel('Grades')
plt.ylabel('Number of Students')
plt.title('Project A')
# plt.text(60, .025, r'$\mu=100,\ \sigma=15$')
plt.axis([0, 100, 0, 30])
plt.show()
