import argparse
import numpy as np
import matplotlib
matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib import rcParams, cycler
from matplotlib.offsetbox import AnchoredText
from sklearn.linear_model import LinearRegression
from linearmodels.iv import IVLIML
import pandas as pd


# Parse command line arguments
parser = argparse.ArgumentParser(
    description="""
    Dynamic Supply and Demand Animation.

    This script animates dynamic supply and demand curves with noise-based perturbations. 
    Users can choose between different noise shock types to see how the supply and demand curves 
    behave and optionally display the estimated demand line based on the intersections.

    Usage examples:
    python supply_demand.py --shock reverse
    python supply_demand.py --shock match --add_px
    python supply_demand.py --shock reverse --add_px
    python supply_demand.py --shock reverse --add_px --speed fast
    """
)

parser.add_argument("--shock", type=str, choices=["reversed", "matched", "independent"], default="matched",
                    help="Type of noise shock for supply and demand curves")
parser.add_argument("--add_px", action="store_true",
                    help="Flag to also add px shock, allowing unbaised estimation of demand")
parser.add_argument("--speed", type=str, choices=["fast", "medium", "slow"], default="medium",
                    help="Speed of the animation")

args = parser.parse_args()

# Run settings
shock = args.shock
add_px = args.add_px
fit_threshold = (50 if add_px else 20)
fit_px_threshold = 100
shock_min_max = (-2.0, 2.0)
px_pct_min_max = (-0.05, 0.05)
animation_interval = (
    50 if add_px and args.speed == "fast"
    else
    100 if add_px and args.speed == "medium"
    else
    200
)
animation_frames = (
    1_000 if add_px and args.speed == "fast"
    else
    500 if add_px and args.speed == "medium"
    else
    250
)

# Function to find intersection
def find_intersection(a_s, b_s, a_d, b_d):
    """
    Find the intersection point of the supply and demand lines.
    
    Parameters:
    a_s (float): Intercept of the supply line
    b_s (float): Slope of the supply line
    a_d (float): Intercept of the demand line
    b_d (float): Slope of the demand line
    
    Returns:
    tuple: Quantity and Price at the intersection point, or (None, None) if no intersection exists
    """
    denominator = (b_s + b_d)
    if denominator == 0:
        return None, None  # Parallel lines, no intersection
    Q_intersect = (a_d - a_s) / denominator
    P_intersect = a_s + b_s * Q_intersect
    return Q_intersect, P_intersect

# Replace the existing style setting with these lines
plt.style.use('ggplot')
plt.rcParams['axes.facecolor'] = '#f0f0f0'
plt.rcParams['figure.facecolor'] = 'white'

# Keep the existing color palette and other custom settings
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
plt.rcParams['axes.prop_cycle'] = cycler(color=colors)

# Custom settings
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
rcParams['font.size'] = 12
rcParams['axes.labelsize'] = 14
rcParams['axes.titlesize'] = 16
rcParams['lines.linewidth'] = 2

# Initialize plot
fig, ax = plt.subplots(figsize=(12, 7))
plt.title(f"Dynamic Supply and Demand Curves ({shock} shocks)", fontweight='bold', pad=20)
plt.xlabel("Quantity", fontweight='bold')
plt.ylabel("Price", fontweight='bold')
ax.set_xlim(0, 10)
ax.set_ylim(0, 25)

# Add a grid to the plot
ax.grid(True, linestyle='--', alpha=0.7)

# Fixed supply and demand slopes
b_s = 1  # Supply slope
b_d = 2  # Demand slope

# Original supply and demand intercepts
a_s_orig = 10  # Original Supply intercept
a_d_orig = 30  # Original Demand intercept

# Original price and quantity intersections
Q_orig, P_orig = find_intersection(a_s_orig, b_s, a_d_orig, b_d)

# Initialize plot lines
Q = np.linspace(0, 20, 400)

# Plot original supply and demand curves
original_supply_y = a_s_orig + b_s * Q
original_demand_y = a_d_orig - b_d * Q
ax.plot(Q, original_supply_y, label="Original Supply", color=colors[0], alpha=0.5, linestyle='--')
ax.plot(Q, original_demand_y, label="Original Demand", color=colors[1], alpha=0.5, linestyle='--')

# Initialize dynamic supply and demand lines
supply_line, = ax.plot([], [], label="Supply + shock", color=colors[0], linestyle="-", alpha=0.8)
demand_line, = ax.plot([], [], label="Demand + shock", color=colors[1], linestyle="-", alpha=0.8)

# Initialize lists to store intersection points
intersection_Q = []
intersection_P = []
biased_elasticity_estimates = []
px_shocks = []
intersection_Qx = []
intersection_Px = []
debiased_elasticity_estimates = []

# Initialize scatter plot for intersections
scatter = ax.scatter([], [], color=colors[2], marker="o", s=50, alpha=0.6, label="Observed P & Q")

# Initialize estimated demand line if enabled
estimated_demand_line, = ax.plot([], [], label="Biased demand estimate", color=colors[3], linestyle="--", linewidth=2)

# Initialize scatter plot for px shocks
if add_px:
    scatter_px = ax.scatter([], [], color=colors[4], marker="x", s=50, alpha=0.6, label="Observed Px and Qx")
    estimated_demand_line_px, = ax.plot([], [], label="Unbiased demand estimate", color=colors[5], linestyle="--", linewidth=2)

# Legend
legend = ax.legend(loc="lower left", framealpha=0.9, facecolor="white", edgecolor="none", fontsize=10)
legend.set_zorder(100)

# Elasticity estimates text box
elasticity_text = AnchoredText("", loc="lower right", frameon=False, prop=dict(size=10))
ax.add_artist(elasticity_text)

# Animation update function
def update(frame):
    """
    Update the supply and demand lines, find the intersection, and update the plot.
    
    Parameters:
    frame (int): The current frame of the animation
    
    Returns:
    tuple: Updated plot elements
    """
    # Generate new intercepts by adding noise to the original intercepts
    noise_a_s = np.random.uniform(*shock_min_max)
    noise_a_d = np.random.uniform(*shock_min_max)
    if shock == "independent":
        pass
    elif shock == "match":
        noise_a_s = -noise_a_d + np.random.uniform(shock_min_max[0]/2, shock_min_max[1]/2)
    elif shock == "reverse":
        noise_a_s = noise_a_d + np.random.uniform(shock_min_max[0]/2, shock_min_max[1]/2)
    else:
        pass
    a_s = a_s_orig + noise_a_s
    a_d = a_d_orig + noise_a_d
    
    # Update supply and demand lines with new intercepts
    supply_y = a_s + b_s * Q
    demand_y = a_d - b_d * Q
    supply_line.set_data(Q, supply_y)
    demand_line.set_data(Q, demand_y)

    # Find new intersection and store values
    Q_i, P_i = find_intersection(a_s, b_s, a_d, b_d)
    intersection_Q.append(Q_i)
    intersection_P.append(P_i)

    # If intersection exists, add it to the scatter plot
    if Q_i is not None:
        scatter.set_offsets(np.c_[intersection_Q, intersection_P])
    
    # If enough points, perform linear regression to estimate new demand depending on the shock
    if (len(intersection_Q) >= fit_threshold):
        X = np.array(intersection_Q).reshape(-1, 1)
        y = np.array(intersection_P)
        model = LinearRegression()
        model.fit(X, y)
        a_est = model.intercept_
        b_est = model.coef_[0]
        biased_elasticity_estimates.append(-P_i/(a_est - P_i))
        estimated_y = a_est + b_est * Q
        estimated_demand_line.set_data(Q, estimated_y)
    else:
        estimated_demand_line.set_data([], [])
    
    # If px is set, calculate and add to scatter plot
    if add_px:
        # Generate random percentage shock between -5% and +5%
        px_shock = P_i * np.random.uniform(*px_pct_min_max)
        
        # Get new prices and intersections
        P_x = P_i + px_shock
        Q_x = (a_d - P_x) / b_d
        
        # Store values
        px_shocks.append(px_shock)
        intersection_Qx.append(Q_x)
        intersection_Px.append(P_x)
        
        # Add to scatter plot if px is set and original intersection exists
        if Q_i is not None:
            scatter_px.set_offsets(np.c_[intersection_Qx, intersection_Px])
    
    if add_px and (len(intersection_Qx) >= fit_px_threshold):
        # Fit an IV model to estimate the unbiased demand curve
        X = np.array(intersection_Qx)
        Z = np.array(px_shocks).reshape(-1, 1)
        y = np.array(intersection_Px).reshape(-1, 1)
        data = pd.DataFrame(np.c_[X, Z, y], columns=["Q", "P_shock", "P"])
        model = IVLIML.from_formula("P ~ 1 + [Q ~ P_shock]", data=data)
        res = model.fit()
        a_est = res.params.iloc[0]
        b_est = res.params.iloc[1]
        debiased_elasticity_estimates.append(-P_i/(a_est - P_i))
        estimated_y = a_est + b_est * Q
        estimated_demand_line_px.set_data(Q, estimated_y)
    elif add_px:
        estimated_demand_line_px.set_data([], [])

    # Update elasticity text
    elasticity_str = f"True Elasticity: {-P_orig/(a_d_orig - P_orig):.2f}"
    if len(biased_elasticity_estimates) > 0:
        elasticity_str += f"\nBiased Estimate: {np.mean(biased_elasticity_estimates[-25:]):.2f}"
    if add_px and len(debiased_elasticity_estimates) > 0:
        elasticity_str += f"\nDebiased Estimate: {np.mean(debiased_elasticity_estimates[-25:]):.2f}"
    elasticity_text.txt.set_text(elasticity_str)

    # Return the updated plot objects
    return_objects = (supply_line, demand_line, scatter, estimated_demand_line, legend, elasticity_text)
    if add_px:
        return_objects += (scatter_px, estimated_demand_line_px)
    return return_objects

# Create animation
ani = FuncAnimation(
    fig,
    update,
    frames=animation_frames,
    interval=animation_interval,
    blit=True,
    repeat=False
)

plt.show()
