import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize

# Define constants
c1 = np.float64(6400)
c2 = np.float64(1483)
l0 = np.float64(0.12156646438729327)
h0 = np.float64(0.08843353561270673)
d = l0 + h0
T = l0 / c1 + h0 / c2
A = (np.square(c1) / np.square(c2)) - 1
C = (np.square(c1) * np.square(T)) - np.square(d)

# These values need to be set by the user
a_l = 1.0  # Example value - replace with your value
b_l = 0.5  # Example value - replace with your value

def equation(alpha, x):
    """
    Returns the value of the equation. 
    When this equals zero, we have a solution.
    """
    K1 = 2 * T * (np.square(c1) / c2)
    K2 = 4 * A * C
    
    # Handle potential numerical issues with the sqrt
    radicand = 2 * d * np.cos(alpha) - K1 - K2
    if radicand < 0:
        return np.inf  # No real solution in this case
    
    term1 = -2 * d * np.cos(alpha) + K1 - np.sqrt(radicand)
    term2 = term1 / (2 * A) * np.cos(alpha)
    
    result = term2 - a_l * x - b_l
    return result

def find_x_for_alpha(alpha):
    """
    For a given alpha, find the corresponding x value
    """
    # Define an equation of one variable (x) with fixed alpha
    def eq_to_solve(x):
        return equation(alpha, x)
    
    # Use a numerical solver to find x
    # Starting with an initial guess of x=0
    try:
        solution = optimize.root_scalar(eq_to_solve, bracket=[-100, 100], method='brentq')
        return solution.root
    except Exception as e:
        # No solution found in the bracket
        return np.nan

def plot_alpha_x_relationship():
    """
    Creates a plot showing the relationship between alpha and x
    """
    # Create an array of alpha values (in radians) from 0 to pi/2
    alpha_values = np.linspace(0.01, np.pi/2 - 0.01, 100)
    
    # Compute corresponding x values
    x_values = []
    for alpha in alpha_values:
        x = find_x_for_alpha(alpha)
        x_values.append(x)
    
    # Plot the relationship
    plt.figure(figsize=(10, 6))
    plt.plot(alpha_values, x_values, 'b-')
    plt.xlabel('Alpha (radians)')
    plt.ylabel('x')
    plt.title('Relationship between alpha and x')
    plt.grid(True)
    plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    plt.axvline(x=0, color='k', linestyle='--', alpha=0.3)
    plt.show()

def find_specific_solution(desired_alpha=None, desired_x=None):
    """
    Find x for a given alpha, or alpha for a given x
    """
    if desired_alpha is not None:
        # Find x for a given alpha
        x = find_x_for_alpha(desired_alpha)
        print(f"For alpha = {desired_alpha} rad ({np.degrees(desired_alpha)}°), x = {x}")
        return x
    
    elif desired_x is not None:
        # Find alpha for a given x using a numerical approach
        def eq_for_alpha(alpha):
            calculated_x = find_x_for_alpha(alpha)
            if np.isnan(calculated_x):
                return np.inf
            return calculated_x - desired_x
        
        try:
            # Try to find alpha in a reasonable range
            solution = optimize.root_scalar(eq_for_alpha, bracket=[0.01, np.pi/2 - 0.01], method='brentq')
            alpha = solution.root
            print(f"For x = {desired_x}, alpha = {alpha} rad ({np.degrees(alpha)}°)")
            return alpha
        except Exception as e:
            print(f"Couldn't find an alpha value for x = {desired_x}. Error: {e}")
            return None
    else:
        print("Please provide either desired_alpha or desired_x")

# Example usage
# Edit a_l and b_l to match your values, then:

def main():
    print("Constants used:")
    print(f"c1 = {c1}, c2 = {c2}")
    print(f"l0 = {l0}, h0 = {h0}")
    print(f"d = {d}, T = {T}")
    print(f"A = {A}, C = {C}")
    print(f"a_l = {a_l}, b_l = {b_l}")
    
    print("\nPlotting the relationship between alpha and x...")
    plot_alpha_x_relationship()
    
    # Find x for a specific alpha (45 degrees)
    find_specific_solution(desired_alpha=np.radians(45))
    
    # Find alpha for a specific x (example value)
    find_specific_solution(desired_x=1.0)

if __name__ == "__main__":
    main()