import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.mixture import GaussianMixture

# Function to randomly sample a subset of attenuation values for each subject (row)
def sample_attenuation(attenuation_list, sample_size=1000):
    """Randomly sample a subset from the attenuation list for each subject."""
    if len(attenuation_list) > sample_size:
        return np.random.choice(attenuation_list, sample_size, replace=False)
    else:
        return attenuation_list
    

def bootstrap_diff_in_means(group_1, group_2, n_iterations=10000):
    """Bootstrap resampling to calculate the distribution of the difference in means."""
    diff_means = []
    for _ in range(n_iterations):
        # Resample with replacement
        sample_1 = np.random.choice(group_1, size=len(group_1), replace=True)
        sample_2 = np.random.choice(group_2, size=len(group_2), replace=True)
        
        # Compute the difference in means
        diff_means.append(np.mean(sample_1) - np.mean(sample_2))
    
    return np.array(diff_means)


def transform_data(data, lamb=0.5, e=1e-6):
    translated_data = data + 1050 + e #make all values positive
    log_transformed = np.log(translated_data)
    sqrt_transformed = np.sqrt(translated_data)
    boxcox_transformed = (translated_data**lamb - 1)/lamb

    _, axes = plt.subplots(2, 2, figsize=(5, 5)) 
    transformations = [
        ('Original Data', translated_data),
        ('Log Transformed', log_transformed),
        ('Square Root Transformed', sqrt_transformed),
        (f'Box-Cox Transformed, lambda={lamb}', boxcox_transformed)
    ]

    for i, (title, transformed_data) in enumerate(transformations):
        ax = axes[i // 2, i % 2] 
        ax.hist(transformed_data, bins=50, color=np.random.rand(3,), alpha=0.7)
        ax.set_title(title)
        ax.set_xlabel('Value')
        ax.set_ylabel('Frequency')

    # Adjust layout to prevent overlapping
    plt.tight_layout()
    plt.show()

    return transformations

def plot_gmm(data):
    gmm = GaussianMixture(n_components=2) 
    gmm.fit(data.reshape(-1, 1))

    x = np.linspace(-1000, 200, 1000).reshape(-1, 1)
    logprob = gmm.score_samples(x) 
    pdf = np.exp(logprob) 

    plt.figure(figsize=(8, 6))
    plt.hist(data, bins=30, density=True, alpha=0.5, color='gray', label='Data histogram')
    plt.plot(x, pdf, label='Fitted GMM', color='red', lw=2)
    plt.title("Gaussian Mixture Model Fit to Data")
    plt.xlabel("Value")
    plt.ylabel("Density")
    plt.legend()
    plt.show()


def compute_aic_and_bic(log_likelihood, n, n_params):
    aic = 2 * n_params - 2 * log_likelihood
    bic = np.log(n) * n_params - 2 * log_likelihood
    
    return aic, bic

def compare_gaussian_models(data, ax, transform=None, lamb=0.5):
    """
    Compares a single Gaussian model with a Gaussian Mixture Model (GMM)
    using AIC and BIC criteria and plots the results on the given axis.
    
    Parameters:
        data (array-like): 1D array-like data to fit the models.
        ax (matplotlib.axis): Axis to plot the results on.
        transform (str): Optional transformation to apply ('log', 'sqrt', 'boxcox').
        lamb (float): Lambda value for Box-Cox transformation (default is 0.5).
        
    Returns:
        None
    """
    if transform is not None:
        # Translate data to make it strictly positive
        data_translated = data - np.min(data) + 1e-6
        
        if transform == 'log':
            data = np.log(data_translated)
        elif transform == 'sqrt':
            data = np.sqrt(data_translated)
        elif transform == 'boxcox':
            data = (data_translated**lamb - 1) / lamb


    # Single Gaussian Model
    mean, std = np.mean(data), np.std(data)
    single_gaussian_pdf = stats.norm.pdf(np.sort(data), loc=mean, scale=std)

    # Fit GMM models with 2 and 3 components
    gmm2 = GaussianMixture(n_components=2, random_state=42)
    gmm2.fit(data.reshape(-1, 1))

    gmm3 = GaussianMixture(n_components=3, random_state=42)
    gmm3.fit(data.reshape(-1, 1))

    # Calculate AIC and BIC for the models
    n = len(data)
    
    log_likelihood_single = np.sum(np.log(single_gaussian_pdf))
    log_likelihood_gmm2 = gmm2.score(data.reshape(-1, 1)) * n
    log_likelihood_gmm3 = gmm3.score(data.reshape(-1, 1)) * n

    aic_single, bic_single = compute_aic_and_bic(log_likelihood_single, n, 2)
    aic_gmm2, bic_gmm2 = compute_aic_and_bic(log_likelihood_gmm2, n, 2 * 2)
    aic_gmm3, bic_gmm3 = compute_aic_and_bic(log_likelihood_gmm3, n, 2 * 3)

    best_aic = min(aic_single, aic_gmm2, aic_gmm3)
    best_bic = min(bic_single, bic_gmm2, bic_gmm3)

    if best_aic == aic_single:
        print(f"Best model according to AIC: Single Gaussian")
    elif best_aic == aic_gmm2:
        print(f"Best model according to AIC: GMM (2 components)")
    else:
        print(f"Best model according to AIC: GMM (3 components)")

    if best_bic == bic_single:
        print(f"Best model according to BIC: Single Gaussian")
    elif best_bic == bic_gmm2:
        print(f"Best model according to BIC: GMM (2 components)")
    else:
        print(f"Best model according to BIC: GMM (3 components)")

    print('\n')

    # Plot results on the provided axis
    x = np.linspace(np.min(data) - 2, np.max(data) + 2, 1000)
    ax.hist(data, bins=30, density=True, alpha=0.5, color='gray', label='Data histogram')
    ax.plot(np.sort(data), single_gaussian_pdf, label='Single Gaussian', color='yellow', lw=2)
    ax.plot(x, np.exp(gmm2.score_samples(x.reshape(-1, 1))), label='2 components', color='red', lw=2)
    ax.plot(x, np.exp(gmm3.score_samples(x.reshape(-1, 1))), label='3 components', color='blue', lw=2)
    ax.set_title("Comparison of Single Gaussian and GMM")
    ax.set_xlabel("Value")
    ax.set_ylabel("Density")
    ax.legend()


def filter_low_density_data(data, density_threshold=0.0005):
    kde = stats.gaussian_kde(data)
    density = kde(data) 
    filtered_data = data[density >= density_threshold]
    return filtered_data

def get_hpdr(arr, hpdr_low=-720, hpdr_high=-300):
    return sum((arr >= hpdr_low) & (arr <= hpdr_high)) / len(arr) * 100