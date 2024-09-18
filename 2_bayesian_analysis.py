import pyro
import pyro.distributions as dist
import torch
from graphviz import Digraph
import pyro.poutine as poutine
import numpy as np
import pdb
import matplotlib.pyplot as plt

# Table of crash causes
crash_cause_fractions = {
    "pilot_error": 0.49,
    "mechanical": 0.23,
    "weather": 0.10,
    "sabotage": 0.08,
    "other": 0.10
}

total_flights = 37.3e6  # Average total flights 2012-2016
total_accidents_per_year = 587  # Average number of accidents per year 2012-2016 (ASN)
fatal_accidents_per_year = 9.8  # Average number of fatal accidents per year 2012-2016 (ASN)

crash_probs = {
    "fatal": fatal_accidents_per_year / total_flights,
    "non_fatal": (total_accidents_per_year - fatal_accidents_per_year) / total_flights
}

crash_cause_probs = {
    "pilot_error": crash_cause_fractions["pilot_error"] * total_accidents_per_year / total_flights,
    "mechanical": crash_cause_fractions["mechanical"] * total_accidents_per_year / total_flights,
    "weather": crash_cause_fractions["weather"] * total_accidents_per_year / total_flights,
    "sabotage": crash_cause_fractions["sabotage"] * total_accidents_per_year / total_flights,
    "other": crash_cause_fractions["other"] * total_accidents_per_year / total_flights,
    "no_crash": 1 - (total_accidents_per_year / total_flights)}
    
# Define the Bayesian Network model
def plane_crash_model(pilot_training=0.0, mcas_system=0.0):
    # Pilot Error influenced by Pilot Training
    # Higher pilot_training leads to higher crash probability (worse training increases error)
    pilot_error_prob = crash_cause_probs["pilot_error"] * (1 + pilot_training)
    pilot_error_prob = min(pilot_error_prob, 1.0)  # Ensure the probability does not exceed 1
    pilot_error = pyro.sample("pilot_error", dist.Bernoulli(pilot_error_prob))
    
    # Mechanical Failure influenced by MCAS system
    # Higher mcas_system leads to higher crash probability (more MCAS issues increase mechanical failures)
    mechanical_failure_prob = crash_cause_probs["mechanical"] * (1 + mcas_system)
    mechanical_failure_prob = min(mechanical_failure_prob, 1.0)  # Ensure the probability does not exceed 1
    mechanical_failure = pyro.sample("mechanical_failure", dist.Bernoulli(mechanical_failure_prob))
    
    # Other causes (Weather, Sabotage, Other)
    weather = pyro.sample("weather", dist.Bernoulli(crash_cause_probs["weather"]))
    sabotage = pyro.sample("sabotage", dist.Bernoulli(crash_cause_probs["sabotage"]))
    other = pyro.sample("other", dist.Bernoulli(crash_cause_probs["other"]))
    no_crash = pyro.sample("no_crash", dist.Bernoulli(crash_cause_probs["no_crash"]))
    
    # Calculate the probability of a plane crash based on the causes
    crash_prob = torch.tensor(pilot_error.item() or mechanical_failure.item() or weather.item() or sabotage.item() or other.item(), dtype=torch.float32)
    
    # Sample plane crash based on crash_prob
    plane_crash = pyro.sample("plane_crash", dist.Bernoulli(crash_prob))
    
    # Fatal crash if a plane crash occurs
    fatal_crash = pyro.sample("fatal_crash", dist.Bernoulli(plane_crash * 0.17))  # 17% chance that a crash is fatal
    # non_fatal_crash = 1 - fatal_crash

    # # Economic cost is very high if a fatal crash occurs
    # non_fatal_cost = 100  # Cost in millions for non-fatal crashes
    # fatal_cost = 3000     # Cost in millions for fatal crashes
    # economic_cost = pyro.sample("economic_cost", dist.Delta(torch.tensor(fatal_crash * fatal_cost + (1 - fatal_crash) * plane_crash * non_fatal_cost)))
            
    # Non-fatal crash is the complement of fatal crash when plane_crash is true
    # non_fatal_crash = plane_crash * (1 - fatal_crash)

    # Economic cost influenced by both fatal and non-fatal crashes
    non_fatal_cost = pyro.sample("non_fatal_cost", dist.Normal(100, 20)) if plane_crash * (1 - fatal_crash) else torch.tensor(0.0)
    fatal_cost = pyro.sample("fatal_cost", dist.Normal(3000, 500)) if fatal_crash else torch.tensor(0.0)
    
    # Total economic cost is based on whether the crash was fatal or non-fatal
    economic_cost = fatal_cost + non_fatal_cost

    return {
        "pilot_error": pilot_error.item(),
        "mechanical_failure": mechanical_failure.item(),
        "weather": weather.item(),
        "sabotage": sabotage.item(),
        "other": other.item(),
        "plane_crash": plane_crash,
        "fatal_crash": fatal_crash.item(),
        "economic_cost": economic_cost.item()
    }

def vectorized_simulation(num_samples, pilot_training=0.0, mcas_system=0.0):
    # Crash cause probabilities
    pilot_error_prob = crash_cause_probs["pilot_error"] * (1 + pilot_training)
    mechanical_failure_prob = crash_cause_probs["mechanical"] * (1 + mcas_system)
    weather_prob = crash_cause_probs["weather"]
    sabotage_prob = crash_cause_probs["sabotage"]
    other_prob = crash_cause_probs["other"]
    
    # Simulate crash causes using Bernoulli trials
    pilot_error = np.random.binomial(1, min(pilot_error_prob, 1.0), num_samples)
    mechanical_failure = np.random.binomial(1, min(mechanical_failure_prob, 1.0), num_samples)
    weather = np.random.binomial(1, weather_prob, num_samples)
    sabotage = np.random.binomial(1, sabotage_prob, num_samples)
    other = np.random.binomial(1, other_prob, num_samples)
    
    # Determine if a plane crash occurred
    plane_crash = (pilot_error | mechanical_failure | weather | sabotage | other)
    
    # Fatal crash occurs in 17% of crashes
    fatal_crash = np.random.binomial(1, 0.17, num_samples) * plane_crash
    
    # Economic costs based on crash type
    non_fatal_costs = np.random.normal(100, 20, num_samples) * (plane_crash & (fatal_crash == 0))
    fatal_costs = np.random.normal(3000, 500, num_samples) * fatal_crash
    
    # Total economic cost is the sum of fatal and non-fatal costs
    economic_cost = non_fatal_costs + fatal_costs
    
    # Return results as a dictionary
    return {
        'pilot_error': pilot_error,
        'mechanical_failure': mechanical_failure,
        'plane_crash': plane_crash,
        'fatal_crash': fatal_crash,
        'economic_cost': economic_cost
    }

# Run the simulation for a number of samples
def run_simulation(num_samples=100, pilot_training=0.5, mcas_system=0.5):
    samples = [plane_crash_model(pilot_training, mcas_system) for _ in range(num_samples)]
    return samples


# Trace the model using Pyro's poutine.trace
def trace_model(model, *args):
    trace = poutine.trace(model).get_trace(*args)
    return trace

# Render the trace using Graphviz
def render_model_with_graphviz(trace):
    dot = Digraph(format='png')
    
    # Iterate through the nodes in the trace
    for name, site in trace.nodes.items():
        if site["type"] == "sample":
            # Add a node for each sample site (e.g., Pilot Error, Mechanical Failure)
            dot.node(name, label=name)

            print(name, site["value"])
            
            # Add connections based on the relationships
            if "pilot_training" in name:
                dot.edge("pilot_training", name)
            if "mcas_system" in name:
                dot.edge("mcas_system", name)
            if name in ["pilot_error", "mechanical_failure", "weather", "sabotage", "other"]:
                dot.edge(name, "plane_crash")
            if name == "plane_crash":
                dot.edge("plane_crash", "fatal_crash")
            if name == "fatal_crash":
                dot.edge("fatal_crash", "economic_cost")
            # if name == "non_fatal_crash":
    
    # Render and save the graph as a PNG file
    dot.render('plane_crash_bayesian_network', view=True)

    # save

# Run the model and trace the execution
trace = trace_model(plane_crash_model, 0.5, 0.5)
render_model_with_graphviz(trace)


# Function to calculate average metrics from vectorized simulations
def calculate_vectorized_metrics(simulations):
    # Total number of samples
    num_samples = len(simulations['plane_crash'])
    
        # Scaling factor: per million flights
    scaling_factor = 1e6 / num_samples
    # Average and standard deviation for economic cost
    avg_cost = np.mean(simulations['economic_cost']) * scaling_factor
    std_cost = np.std(simulations['economic_cost']) * scaling_factor
    
    # Total crashes and fatal crashes
    total_crashes = np.sum(simulations['plane_crash']) * scaling_factor
    total_fatal_crashes = np.sum(simulations['fatal_crash']) * scaling_factor
    
    # Probability of crash and fatal crash (as fractions of the total samples)
    crash_prob = total_crashes / num_samples * scaling_factor
    fatal_crash_prob = total_fatal_crashes / num_samples * scaling_factor
    
    # Standard deviation for crash and fatal crash probabilities
    std_crash_prob = np.std(simulations['plane_crash']) * scaling_factor
    std_fatal_crash_prob = np.std(simulations['fatal_crash']) * scaling_factor
    
    return {
        'total_cost': np.sum(simulations['economic_cost']),
        'avg_cost': avg_cost,
        'std_avg_cost': std_cost,
        'total_crashes': total_crashes,
        'total_fatal_crashes': total_fatal_crashes,
        'crash_probability': crash_prob,
        'std_crash_probability': std_crash_prob,
        'fatal_crash_probability': fatal_crash_prob,
        'std_fatal_crash_probability': std_fatal_crash_prob
    }

# Function to plot crashes per million flights with error bars for multiple simulations
def plot_crashes_per_million_flights_with_error(list_of_simulations, labels, title_prefix=""):
    # Initialize lists to store crashes per million and standard errors
    crashes_per_million_list = []
    standard_error_list = []

    # Loop through each set of simulations
    for simulations in list_of_simulations:
        crashes = simulations['plane_crash']

        # Total number of flights
        total_flights = len(crashes)

        # Calculate crashes per million flights
        crashes_per_million = (np.sum(crashes) / total_flights) * 1e6
        crashes_per_million_list.append(crashes_per_million)
        
        # Calculate the proportion of crashes (p) and standard error (SE)
        p_crash = np.sum(crashes) / total_flights
        standard_error = np.sqrt(p_crash * (1 - p_crash) / total_flights) * 1e6  # SE scaled to per million flights
        standard_error_list.append(standard_error)

    # Plot the bar chart with error bars
    plt.figure(figsize=(10, 6))
    plt.bar(labels, crashes_per_million_list, yerr=standard_error_list, capsize=10, color = ['mediumseagreen', 'lightcoral'])
    plt.title(f'{title_prefix}Crashes per Million Flights')
    plt.ylabel('Number of Crashes')
    plt.xlabel('Simulations')
    
    # Save plot
    plt.savefig('crashes_per_million_flights_with_error.pdf', format='pdf', bbox_inches='tight', dpi=300)

def create_latex_table(metrics, metrics_mcas):
    # Start LaTeX table
    latex_table = r"""
    \begin{table}[ht]
    \centering
    \begin{tabular}{lcc}
    \hline
    \textbf{Metric} & \textbf{Without MCAS (Mean $\pm$ Std)} & \textbf{With MCAS (Mean $\pm$ Std)} \\
    \hline
    """
    
    # List of metrics and their LaTeX-friendly names
    metric_names = {
        'avg_cost': 'Average Cost (in millions)',
        'total_crashes': 'Total Crashes',
        'total_fatal_crashes': 'Total Fatal Crashes',
        'crash_probability': 'Crash Probability',
        'fatal_crash_probability': 'Fatal Crash Probability'
    }
    
    # Function to format values in scientific notation with 2 decimals, curly brackets around the exponent
    # If the power is 0, it omits the power of 10 entirely
    def format_scientific(value):
        if isinstance(value, (int, float)):
            if value == 0:
                return f"0.00"
            else:
                # Use scientific notation with two decimals
                formatted_value = f"{value:.2e}"
                base, exponent = formatted_value.split("e")
                exponent = int(exponent)  # Convert the exponent to an integer
                if exponent == 0:
                    return f"{float(base):.2f}"  # Return only the base if the exponent is 0
                else:
                    return f"{float(base):.2f} \\times 10^{{{exponent}}}"  # Include the exponent with curly brackets
        return value
    
    # Add rows for each metric
    for key, name in metric_names.items():
        # Get mean and std for each metric and format them
        mean_without_mcas = metrics.get(key, ' ')
        std_without_mcas = metrics.get(f'std_{key}', ' ')
        mean_with_mcas = metrics_mcas.get(key, ' ')
        std_with_mcas = metrics_mcas.get(f'std_{key}', ' ')

        # Format values using the scientific notation function
        mean_without_mcas_str = format_scientific(mean_without_mcas)
        std_without_mcas_str = format_scientific(std_without_mcas)
        mean_with_mcas_str = format_scientific(mean_with_mcas)
        std_with_mcas_str = format_scientific(std_with_mcas)
        
        # Add row for each metric, with mean and std in the same cell
        latex_table += f"{name} & {mean_without_mcas_str} $\\pm$ {std_without_mcas_str} & {mean_with_mcas_str} $\\pm$ {std_with_mcas_str} \\\\\n"

    # End table
    latex_table += r"""
    \hline
    \end{tabular}
    \caption{Comparison of Metrics Without MCAS and With MCAS}
    \label{tab:comparison_metrics}
    \end{table}
    """
    
    # Print the LaTeX table directly
    print(latex_table)

# Function to plot sensitivity results
def plot_sensitivity(results, param_name, y_label, metric_name):
    param_values = [result['param_value'] for result in results]
    metric_values = [result[metric_name] for result in results]
    
    plt.clf()
    plt.plot(param_values, metric_values, marker='o')
    plt.xlabel(f'{param_name.capitalize()}')
    plt.ylabel(f'{y_label}')
    plt.title(f'Sensitivity of {y_label} to {param_name.capitalize()}')
    plt.grid(True)
    plt.savefig(f'sensitivity_{param_name}_{metric_name}.pdf', format='pdf', bbox_inches='tight', dpi=300)

# Function to run sensitivity analysis for a given parameter
def sensitivity_analysis(vectorized_simulation_func, param_name, param_range, fixed_params, num_samples=10_000_000):
    results = []
    
    for param_value in param_range:
        # Set the current parameter value
        current_params = fixed_params.copy()
        current_params[param_name] = param_value
        
        # Run the simulation with the varying parameter
        simulations = vectorized_simulation_func(num_samples, **current_params)
        
        # Collect crash probability and average economic cost
        crash_probability = np.mean(simulations['plane_crash'])
        avg_economic_cost = np.mean(simulations['economic_cost'])
        
        # Append results (you can track other metrics here if needed)
        results.append({
            'param_value': param_value,
            'crash_probability': crash_probability,
            'avg_economic_cost': avg_economic_cost
        })
    
    return results

if __name__ == "__main__":
    simulations = vectorized_simulation(10_000_000, pilot_training=0.0, mcas_system=0.0)
    metrics = calculate_vectorized_metrics(simulations)
    
    # Assume likelihood of the original probability for pilot pilot errror is 0.5 and mechanical failure due to mcas system is 0.2
    simulations_mcas = vectorized_simulation(10_000_000, pilot_training=0.5, mcas_system=0.2)
    metrics_mcas = calculate_vectorized_metrics(simulations_mcas)

    simulations_list = [simulations, simulations_mcas]
    labels = ['Baseline', 'With MCAS']
    plot_crashes_per_million_flights_with_error(simulations_list, labels, title_prefix="MCAS System ")
    create_latex_table(metrics, metrics_mcas)

    # Example of running sensitivity analysis for pilot training
    pilot_training_range = np.linspace(0.0, 1.0, 10)  # Range of pilot training values from 0.0 to 1.0
    fixed_params = {'pilot_training': 0.0, 'mcas_system': 0.2}  # Keep MCAS system constant

    # Run sensitivity analysis for pilot training
    pilot_training_sensitivity_results = sensitivity_analysis(vectorized_simulation, 'pilot_training', pilot_training_range, fixed_params)

    # Example of running sensitivity analysis for MCAS system failure probability
    mcas_system_range = np.linspace(0.0, 1.0, 10)  # Range of MCAS system reliability from 0.0 to 1.0
    fixed_params_mcas = {'pilot_training': 0.5, 'mcas_system': 0.0}  # Keep pilot training constant

    # Run sensitivity analysis for MCAS system
    mcas_system_sensitivity_results = sensitivity_analysis(vectorized_simulation, 'mcas_system', mcas_system_range, fixed_params_mcas)


    # Plot the sensitivity results for crash probability (pilot training)
    plot_sensitivity(pilot_training_sensitivity_results, 'pilot_training', 'Crash Probability', 'crash_probability')

    # Plot the sensitivity results for average economic cost (pilot training)
    plot_sensitivity(pilot_training_sensitivity_results, 'pilot_training', 'Average Economic Cost (in millions)', 'avg_economic_cost')

    # Plot the sensitivity results for crash probability (MCAS system)
    plot_sensitivity(mcas_system_sensitivity_results, 'mcas_system', 'Crash Probability', 'crash_probability')

    # Plot the sensitivity results for average economic cost (MCAS system)
    plot_sensitivity(mcas_system_sensitivity_results, 'mcas_system', 'Average Economic Cost (in millions)', 'avg_economic_cost')
