#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(5)
fruit = np.random.randint(0, 20, (4, 3))
peps = ['Farrah', 'Fred', 'Felicia']
fruit_names = ['apples', 'bananas', 'oranges', 'peaches']
colors = ['red', 'yellow', '#ff8000', '#ffe5b4']
width = 0.5       # the width of the bars

fig, ax = plt.subplots()

# stacking fruits :D
height = np.zeros(3)
for idx, fruit_name in enumerate(fruit_names):
    ax.bar(peps,
           fruit[idx], width,
           label=fruit_name,
           color=colors[idx],
           bottom=height
           )
    height += fruit[idx]

ax.set_ylabel('Quantity of Fruit')
ax.set_title('Number of Fruit per Person')
ax.legend()
plt.yticks(np.arange(0, 80, step=10))
plt.show()
