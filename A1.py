# copy the code into the ques. jupiter notebook!!!!!!!!!!

import numpy as np
import math
import np.array
import matplotlib.pyplot as plt

# Question 1: Area of triangle
# Q(1.1)
def triangle_area_heron(a, b, c):
    '''
   Compute the triangle area using Heron's formula

   Input: a, b, c (side lengths of triangle)
   Output: area of the triangle (Heron's formula)
    '''
    s = (a+b+c)/2
    area = np.sqrt(s*(s-a)*(s-b)*(s-c))

    return area

# Q(1.2) - Alternative area of triangle formula 
def triangle_area_kahan(a, b, c):
    '''
   Compute the triangle area using Kahan's formula

   Input: a, b, c (side lengths of triangle)
   Output: area of the triangle (Kahan's formula)
    '''
    a, b, c = sorted([a, b, c], reverse=True)  # Ensure a >= b >= c
    if a >= b >= c:
        area = np.sqrt((a+(b+c))*(c-(a-b)*(c+(a-b)*(a+(b-c)))))/4

        return area
    
    else: #raise error message
        raise ValueError("Please input a new input a, b, c such that a >= b >= c.")

# Q(1.3) - Consider different values of z:
# z is some positive number (>= 0)
a = 2 * 𝜀
b = c = np.sqrt(1+(𝜀 ** 4))/𝜀
heron_areas.append(triangle_area_heron(a, b, c))
kahan_areas.append(triangle_area_kahan(a, b, c))

# Convert to arrays for easier analysis
heron_areas = np.array(heron_areas)
kahan_areas = np.array(kahan_areas)

# Generate values for epsilon and compute areas
epsilon_values = np.logspace(-8, 0, 10)  # Log-spaced values from 1e-8 to 1
heron_areas = []
kahan_areas = []

# Compute absolute differences to compare accuracy
errors = np.abs(heron_areas - kahan_areas)

# Plot results
plt.figure(figsize=(8, 6))
plt.loglog(epsilon_values, heron_areas, 'o-', label="Heron's Formula")
plt.loglog(epsilon_values, kahan_areas, 's-', label="Kahan's Formula")
plt.xlabel("Epsilon")
plt.ylabel("Triangle Area")
plt.title("Comparison of Heron's and Kahan's Formulas")
plt.legend()
plt.grid(True, which="both", linestyle="--", linewidth=0.5)
plt.show()

# Plot absolute errors
plt.figure(figsize=(8, 6))
plt.loglog(epsilon_values, errors, 'o-', color='r', label="Absolute Error")
plt.xlabel("Epsilon")
plt.ylabel("|Heron - Kahan|")
plt.title("Error between Heron's and Kahan's Formulas")
plt.legend()
plt.grid(True, which="both", linestyle="--", linewidth=0.5)
plt.show()

# Discussion Summary - 300 Words
# The absolute difference between the area after performing Heron's and Kahan's formula for different values of 𝜀 are on order 10e-16 to 10e-17 (close to machine precision)
# This can be implied that for these ranges of 𝜀, both methods produce nearly identical results
# When 𝜀 increases, the error tend to increase slightly which will accumulate small numerical errors
# Kahan's doesn't show significant improvement over Heron's (the triangle is not extremely ill-conditioned for small 𝜀)


# Question 2: Numerical Linear Algebra
# Q(2.1) - accept input n[int], for loop - infinite
def sequence_element(n):
    '''
   Compute and return x[n+1], which is the matrix product of matrix A ad the previous matrix x[n]

   Input: n (non-negative integer)
   Output: return x[n] with integer scalar data type, shape(2,)
    '''
if n = int and n >= 0:

    # define x[0]
    x[0]  = np.array([[1],
                      [1]], dtype=int)
    
    # define A
    A = np.array([[0,1],
                  [1,1]], dtype=int)
    
    for n in range(n):
         x = np.dot(A, x) # Matrix-vector multiplication
    
    return x

else:
    raise ValueError("Please input non-negative n.")


# Q(2.2) - Perform numerical calculations to investigate
def investigate_error(n_values):
    '''
   Compute e[n] according the formula given in the question

   Input: n_values (integer)
   Output: e[n] value
    '''
#trial with different values of n

# set the scene
def seq_element(n):
    x[0]  = np.array([[1],
                      [1]], dtype=int)
    A = np.array([[0,1],
                  [1,1]], dtype=int)
    for n in range(n):
        x = np.dot(A, x) # Matrix-vector multiplication
    return x

def compute_ratios(N_max):
    A = np.array([[0,1],
                  [1,1]], dtype=int)
   
    eigenvalues = np.linalg.eig(A) # eigenvalue of A
    alpha = max(abs(eigenvalues))  # Largest eigenvalue by magnitude
    ratios = []
    ns = list(range(1, N_max + 1))

for n in ns:
    x_n = sequence_element(n)
    # compute e[n]
    ratio = np.linalg.norm(A * x_n - alpha * x_n) / np.linalg.norm(x_n)
    ratios.append(ratio)

# Define range of n valued
N_max = 20
ns, ratios = compute_ratios(N_max)

# Plot the results
plt.plot(ns, ratios, marker='o', linestyle='-')
plt.xlabel("n")
plt.ylabel("Error ratio")
plt.title("Convergence of A * x_n - alpha * x_n/x_n")
plt.grid()
plt.show()

# Discussion Summary - 300 Words

# Q(3.1) 
def interpolatory_quadrature_weights(x):
    N = len(x) - 1  #len(x) = N + 1
    w = np.zeros(N + 1)
    for i in range(N + 1):
        L_i = np.ones_like(x)
        for j in range(N + 1):
            if i != j:
                # finding the interpolating polynomial
                L_i *= (x - x[j]) / (x[i] - x[j])
        w[i] = np.trapz(L_i, x) # trapezoid rule of polynomial and x
    return w

# Find the interpolating polynomial coefficients (degree N-1)
p_coeffs = np.polyfit(xi, f(xi), N) # polynomial degree(N-1)

# Evaluate the interpolationg polynomial using these coefficients, to plot it smoothly
p_plot = np.polyval(p_coeffs, x_plot) #!!!!!!!!!

# Test the function with known quadrature rules
# Midpoint Rule (N=0)
xi_mid = np.array([0.0])
w_mid = interpolatory_quadrature_weights(xi_mid)
print("Midpoint rule weights:", w_mid)

# Trapezoidal Rule (N=1)
xi_trap = np.array([-1.0, 1.0])
w_trap = interpolatory_quadrature_weights(xi_trap)
print("Trapezoidal rule weights:", w_trap)

# Simpson's Rule (N=2)
xi_simp = np.array([-1.0, 0.0, 1.0])
w_simp = interpolatory_quadrature_weights(xi_simp)
print("Simpson's rule weights:", w_simp)



# Q(3.2)
# Define the function f(x)
def f(x):
    return 1 / (1 + (3 * x) ** 2)

# Composite quadrature using equally spaced nodes (x0)
def composite_midpoint_rule(N):
    x0 = np.linspace(-1, 1, N + 1)
    h = 2 / N
    integral = h * np.sum(f((x0[:-1] + x0[1:]) / 2))
    return integral

# Clenshaw-Curtis quadrature using Chebyshev nodes (x1)
def clenshaw_curtis_rule(N):
    x1 = -np.cos(np.pi * np.arange(N + 1) / N)
    w = np.zeros(N + 1)
    for k in range(0, N + 1, 2):
        w += (2 / (1 - k ** 2)) * np.cos(k * np.pi * np.arange(N + 1) / N)
    w[0] /= 2
    w[-1] /= 2
    integral = np.sum(w * f(x1))
    return integral

# Compute exact integral using numerical integration
exact_integral = 2 / (3 * np.arctan(3))

# Compare accuracy for different N values
N_values = np.arange(2, 50, 2)
errors_midpoint = []
errors_clenshaw = []

for N in N_values:
    I_midpoint = composite_midpoint_rule(N)
    I_clenshaw = clenshaw_curtis_rule(N)
    errors_midpoint.append(abs(I_midpoint - exact_integral))
    errors_clenshaw.append(abs(I_clenshaw - exact_integral))

# Plot errors
plt.figure(figsize=(8, 6))
plt.semilogy(N_values, errors_midpoint, label='Composite Midpoint Rule', marker='o')
plt.semilogy(N_values, errors_clenshaw, label='Clenshaw-Curtis Rule', marker='s')
plt.xlabel('N')
plt.ylabel('Absolute Error')
plt.title('Error Comparison of Quadrature Rules')
plt.legend()
plt.grid()
plt.show()

# Discussion Summary - 300 Words
