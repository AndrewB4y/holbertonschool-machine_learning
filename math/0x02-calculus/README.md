# 0x02-calculus

A project created to work and study calculus with python

## Tasks
0. Sigma is for Sum
$\sum_{i=2}^{5} i

1. 3 + 4 + 5
2. 3 + 4
3. 2 + 3 + 4 + 5
4. 2 + 3 + 4

Ans:/0-sigma_is_for_sum
  
1. The Greeks pronounce it sEEgma
\sum_{k=1}^{4} 9i - 2k

90 - 20
36i - 20
90 - 8k
36i - 8k
Repo:

GitHub repository: holbertonschool-machine_learning
Directory: math/0x02-calculus
File: 1-seegma
  
2. Pi is for Product
mandatory
\prod_{i = 1}^{m} i

(m - 1)!
0
(m + 1)!
m!
Repo:

GitHub repository: holbertonschool-machine_learning
Directory: math/0x02-calculus
File: 2-pi_is_for_product
  
3. The Greeks pronounce it pEE
mandatory
\prod_{i = 0}^{10} i

10!
9!
100
0
Repo:

GitHub repository: holbertonschool-machine_learning
Directory: math/0x02-calculus
File: 3-pee
  
4. Hello, derivatives!
mandatory
\frac{dy}{dx} where y = x^4 + 3x^3 - 5x + 1

3x^3 + 6x^2 -4
4x^3 + 6x^2 - 5
4x^3 + 9x^2 - 5
4x^3 + 9x^2 - 4
Repo:

GitHub repository: holbertonschool-machine_learning
Directory: math/0x02-calculus
File: 4-hello_derivatives
  
5. A log on the fire
mandatory
\frac{d (xln(x))}{dx}

ln(x)
\frac{1}{x} + 1
ln(x) + 1
\frac{1}{x}
Repo:

GitHub repository: holbertonschool-machine_learning
Directory: math/0x02-calculus
File: 5-log_on_fire
  
6. It is difficult to free fools from the chains they revere
mandatory
\frac{d (ln(x^2))}{dx}

\frac{2}{x}
\frac{1}{x^2}
\frac{2}{x^2}
\frac{1}{x}
Repo:

GitHub repository: holbertonschool-machine_learning
Directory: math/0x02-calculus
File: 6-voltaire
  
7. Partial truths are often more insidious than total falsehoods
mandatory
\frac{\partial f(x, y)}{\partial y} where f(x, y) = e^{xy} and \frac{\partial&space;x}{\partial&space;y}=\frac{\partial&space;y}{\partial&space;x}=0

e^{xy}
ye^{xy}
xe^{xy}
e^{x}
Repo:

GitHub repository: holbertonschool-machine_learning
Directory: math/0x02-calculus
File: 7-partial_truths
  
8. Put it all together and what do you get?
mandatory
\frac{\partial^2}{\partial&space;y\partial&space;x}(e^{x^2y}) where \frac{\partial&space;x}{\partial&space;y}=\frac{\partial&space;y}{\partial&space;x}=0

2x(1+y)e^{x^2y}
2xe^{2x}
2x(1+x^2y)e^{x^2y}
e^{2x}
Repo:

GitHub repository: holbertonschool-machine_learning
Directory: math/0x02-calculus
File: 8-all-together
  
9. Our life is the sum total of all the decisions we make every day, and those decisions are determined by our priorities
mandatory
Write a function def summation_i_squared(n): that calculates \sum_{i=1}^{n} i^2:

n is the stopping condition
Return the integer value of the sum
If n is not a valid number, return None
You are not allowed to use any loops
alexa@ubuntu:0x02-calculus$ cat 9-main.py 
#!/usr/bin/env python3

summation_i_squared = __import__('9-sum_total').summation_i_squared

n = 5
print(summation_i_squared(n))
alexa@ubuntu:0x02-calculus$ ./9-main.py 
55
alexa@ubuntu:0x02-calculus$
Repo:

GitHub repository: holbertonschool-machine_learning
Directory: math/0x02-calculus
File: 9-sum_total.py
  
10. Derive happiness in oneself from a good day's work
mandatory
Write a function def poly_derivative(poly): that calculates the derivative of a polynomial:

poly is a list of coefficients representing a polynomial
the index of the list represents the power of x that the coefficient belongs to
Example: if f(x) = x^3 + 3x +5, poly is equal to [5, 3, 0, 1]
If poly is not valid, return None
If the derivative is 0, return [0]
Return a new list of coefficients representing the derivative of the polynomial
alexa@ubuntu:0x02-calculus$ cat 10-main.py 
#!/usr/bin/env python3

poly_derivative = __import__('10-matisse').poly_derivative

poly = [5, 3, 0, 1]
print(poly_derivative(poly))
alexa@ubuntu:0x02-calculus$ ./10-main.py 
[3, 0, 3]
alexa@ubuntu:0x02-calculus$
Repo:

GitHub repository: holbertonschool-machine_learning
Directory: math/0x02-calculus
File: 10-matisse.py
  
11. Good grooming is integral and impeccable style is a must
mandatory


3x2 + C
x4/4 + C
x4 + C
x4/3 + C
Repo:

GitHub repository: holbertonschool-machine_learning
Directory: math/0x02-calculus
File: 11-integral
  
12. We are all an integral part of the web of life
mandatory


e2y + C
ey + C
e2y/2 + C
ey/2 + C
Repo:

GitHub repository: holbertonschool-machine_learning
Directory: math/0x02-calculus
File: 12-integral
  
13. Create a definite plan for carrying out your desire and begin at once
mandatory


3
6
9
27
Repo:

GitHub repository: holbertonschool-machine_learning
Directory: math/0x02-calculus
File: 13-definite
  
14. My talents fall within definite limitations
mandatory


-1
0
1
undefined
Repo:

GitHub repository: holbertonschool-machine_learning
Directory: math/0x02-calculus
File: 14-definite
  
15. Winners are people with definite purpose in life
mandatory


5
5x
25
25x
Repo:

GitHub repository: holbertonschool-machine_learning
Directory: math/0x02-calculus
File: 15-definite
  
16. Double whammy
mandatory


9ln(2)
9
27ln(2)
27
Repo:

GitHub repository: holbertonschool-machine_learning
Directory: math/0x02-calculus
File: 16-double