import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter, binary_dilation
from scipy.optimize import differential_evolution
from scipy.ndimage import grey_dilation, grey_erosion, generate_binary_structure, distance_transform_edt
import logging



default_box_size = 60

def find_peak_and_trough_with_center_adjustment(
    image_array, y, x, max_distance, num_angles=360,
    num_samples=100, delta=2, center_steps=5):
    """
    Finds the peak and trough positions along a line rotating around the center,
    allowing for small adjustments to the center position.

    Parameters:
    - image_array: 2D numpy array representing the image.
    - y, x: Initial center coordinates.
    - max_distance: Maximum length of the line.
    - num_angles: Number of angles to consider between 0 and 2π.
    - num_samples: Number of points to sample along each line.
    - delta: Maximum shift allowed for the center coordinates.
    - center_steps: Number of steps in each direction for center adjustment.

    Returns:
    - best_center: Tuple of new center coordinates (y, x).
    - high_peak_pos: Position of the peak (y, x).
    - low_peak_pos: Position of the trough (y, x).
    - best_angle: Angle at which maximum contrast occurs.
    """
    max_contrast = -np.inf
    best_angle = None  # Initialize best_angle
    best_center = (y, x)  # Initialize best_center
    high_peak_pos = None
    low_peak_pos = None

    # Generate center adjustments
    dy_range = np.linspace(-delta, delta, center_steps)
    dx_range = np.linspace(-delta, delta, center_steps)

    for dy in dy_range:
        for dx in dx_range:
            adjusted_y = y + dy
            adjusted_x = x + dx

            # Ensure the adjusted center is within image bounds
            if not (0 <= adjusted_y < image_array.shape[0] and
                    0 <= adjusted_x < image_array.shape[1]):
                continue

            # Perform the line rotation and contrast computation
            for angle in np.linspace(0, 2*np.pi, num_angles, endpoint=False):
                # Initialize variables to store max and min values along the line
                max_value = -np.inf
                min_value = np.inf
                max_pos = None
                min_pos = None

                # Generate points along the line at the current angle
                for t in np.linspace(-max_distance/2, max_distance/2, num_samples):
                    ny = adjusted_y + t * np.sin(angle)
                    nx = adjusted_x + t * np.cos(angle)

                    # Round to nearest integer pixel coordinates
                    ny_int = int(round(ny))
                    nx_int = int(round(nx))

                    # Check if the coordinates are within image bounds
                    if (0 <= ny_int < image_array.shape[0] and
                        0 <= nx_int < image_array.shape[1]):
                        value = image_array[ny_int, nx_int]

                        # Update max and min values and their positions
                        if value > max_value:
                            max_value = value
                            max_pos = (ny_int, nx_int)
                        if value < min_value:
                            min_value = value
                            min_pos = (ny_int, nx_int)

                # Skip if no valid points were found along the line
                if max_value == -np.inf or min_value == np.inf:
                    continue

                # Calculate contrast for the current angle
                contrast = max_value - min_value

                # Update the best results if a higher contrast is found
                if contrast > max_contrast:
                    max_contrast = contrast
                    best_angle = angle
                    high_peak_pos = max_pos
                    low_peak_pos = min_pos
                    best_center = (adjusted_y, adjusted_x)

    # Check if best_angle was assigned
    if best_angle is None:
        # Handle the case where no valid contrast was found
        print("Warning: No valid contrast found; returning initial center and angle 0.")
        best_angle = 0  # Assign a default value
        high_peak_pos = (int(round(y)), int(round(x)))
        low_peak_pos = (int(round(y)), int(round(x)))
        best_center = (y, x)

    # Adjust angle to be between -pi and pi
    best_angle = (best_angle + np.pi) % (2 * np.pi) - np.pi

    return best_center, high_peak_pos, low_peak_pos, best_angle




# Rotation and transformation functions
def rotate_x(coordinates, theta):
    x, y, z = coordinates
    y_rot = y * np.cos(theta) - z * np.sin(theta)
    z_rot = y * np.sin(theta) + z * np.cos(theta)
    return x, y_rot, z_rot

def rotate_y(coordinates, theta):
    x, y, z = coordinates
    x_rot = x * np.cos(theta) + z * np.sin(theta)
    z_rot = -x * np.sin(theta) + z * np.cos(theta)
    return x_rot, y, z_rot

def rotate_z(coordinates, theta):
    x, y, z = coordinates
    x_rot = x * np.cos(theta) - y * np.sin(theta)
    y_rot = x * np.sin(theta) + y * np.cos(theta)
    return x_rot, y_rot, z

def translate(x, y, translation_factor_x, translation_factor_y):
    x_translated = x - translation_factor_x
    y_translated = y - translation_factor_y
    return x_translated, y_translated

def shear(x, y, shear_factor_x, shear_factor_y):
    x_sheared = x + shear_factor_x * y
    y_sheared = y + shear_factor_y * x
    return x_sheared, y_sheared

def stretch(x, y, stretch_factor_x, stretch_factor_y):
    x_stretched = x * stretch_factor_x
    y_stretched = y * stretch_factor_y
    return x_stretched, y_stretched

def gaussian(x, y, amplitude, height_factor, baseline):
    z = amplitude * (x + height_factor) * np.exp(-( (x) ** 2 + (y) ** 2)) + baseline
    return z

def model(xy, translation_factor_x, translation_factor_y, stretch_factor_x, stretch_factor_y, theta_x, theta_y, theta_z, amplitude, height_factor, baseline, shear_factor_x, shear_factor_y):
    x, y = xy
    x, y = translate(x, y, translation_factor_x, translation_factor_y)
    x, y, _ = rotate_z((x, y, 0), theta_z)
    x, y = shear(x, y, shear_factor_x, shear_factor_y)
    x, y = stretch(x, y, stretch_factor_x, stretch_factor_y)
    z = gaussian(x, y, amplitude, height_factor, baseline)
    return z

def min_max_normalize(data):
    """Normalize a NumPy array to a [0, 1] range."""
    min_val = np.min(data)
    max_val = np.max(data)
    normalized_data = (data - min_val) / (max_val - min_val)
    return normalized_data
    
# Define the combined model
def combined_model(xy, *params):
    x, y = xy
    n_params_per_dislocation = 12  # Number of parameters per dislocation
    n_dislocations = len(params) // n_params_per_dislocation
    z_total = np.zeros_like(x)
    for i in range(n_dislocations):
        p = params[i*n_params_per_dislocation:(i+1)*n_params_per_dislocation]
        z_total += model(xy, *p)
    return z_total


def fit_multiple_dislocations(patch, centers, bounds, patch_bounds, fit=True):
    """
    Fit multiple dislocations within a given patch.

    Parameters:
    - patch (2D array): The image patch containing dislocations.
    - centers (list of [x, y]): Initial centers of the dislocations.
    - bounds (list of [min_x, min_y, max_x, max_y]): Bounds for each dislocation.
    - patch_bounds (list or tuple): Bounds of the entire patch.
    - fit (bool): If True, perform fitting. If False, return initial guesses.

    Returns:
    - popt (array or list): Optimized parameters if fit=True, else initial guesses.
    - None
    - xy (2xN array): Meshgrid coordinates flattened.
    - centers_of_fits_full_image_coordinates (list of tuples): Fitted centers in full image coordinates.
    - pixel_size (float): Pixel size based on patch dimensions.
    """
    
    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s:%(message)s')

    best_center_pos = []
    high_peak_pos = []
    low_peak_pos = []
    best_angle = []
    std_dev = 2
    num_points = 30
    weights = np.zeros_like(patch, dtype=float)
    gradient_y, gradient_x = np.gradient(patch)
    gradient_magnitude = np.hypot(gradient_x, gradient_y)

    # Adjust bounds and centers by patch_bounds
    for i in range(len(bounds)):
        bounds[i] = [
            bounds[i][0] - patch_bounds[0],
            bounds[i][1] - patch_bounds[1],
            bounds[i][2] - patch_bounds[0],
            bounds[i][3] - patch_bounds[1]
        ]
        centers[i] = [
            centers[i][0] - patch_bounds[0],
            centers[i][1] - patch_bounds[1]
        ]

    # Lists to store translation guesses and bounds for plotting
    translation_x_guesses = []
    translation_y_guesses = []
    translation_bounds_x_list = []
    translation_bounds_y_list = []
    peak_x_list = []
    peak_y_list = []
    valid_centers = []
    valid_bounds = []

    for i in range(len(centers)):
        min_x, min_y, max_x, max_y = bounds[i]
        x_center, y_center = centers[i]
        largest_side = max(max_x - min_x, max_y - min_y)

        # Find the peak of the gradient
        local_gradient = gradient_magnitude[min_y:max_y, min_x:max_x]
        if local_gradient.size == 0:
            logging.warning(f"No gradient data in bounds for dislocation {i}. Skipping.")
            continue

        peak_y_local, peak_x_local = np.unravel_index(
            np.argmax(local_gradient, axis=None),
            local_gradient.shape
        )
        peak_x = peak_x_local + min_x
        peak_y = peak_y_local + min_y
        peak_x_list.append(peak_x)
        peak_y_list.append(peak_y)

        # Append to valid lists
        valid_centers.append(centers[i])
        valid_bounds.append(bounds[i])

        # Find the best center and angle
        best_center, high_peak, low_peak, angle = find_peak_and_trough_with_center_adjustment(
            patch, peak_y, peak_x, largest_side // 2
        )
        best_center_pos.append(best_center)
        high_peak_pos.append(high_peak)
        low_peak_pos.append(low_peak)
        best_angle.append(angle)
        x_center, y_center = best_center

        # Create the mask made up of Gaussians along a Gaussian line
        weight_radius = int((max_x - min_x + max_y - min_y) / 2)
        current_length = np.hypot(high_peak[0] - low_peak[0], high_peak[1] - low_peak[1])
        if current_length == 0:
            logging.warning(f"Zero length between high and low peaks for dislocation {i}. Skipping.")
            continue  # Avoid division by zero

        extended_length = current_length * 2.5
        direction_vector_y = (high_peak[0] - low_peak[0]) / current_length
        direction_vector_x = (high_peak[1] - low_peak[1]) / current_length
        new_high_peak = (
            low_peak[0] + extended_length * direction_vector_y,
            low_peak[1] + extended_length * direction_vector_x
        )
        new_low_peak = (
            high_peak[0] - extended_length * direction_vector_y,
            high_peak[1] - extended_length * direction_vector_x
        )
        x_line = np.linspace(new_low_peak[1], new_high_peak[1], num_points)
        y_line = np.linspace(new_low_peak[0], new_high_peak[0], num_points)
        z_line = np.exp(-np.linspace(-4, 4, num_points) ** 2 / (2 * std_dev ** 2))
        z_line = np.clip(z_line, 0, 0.7)
        z_line = z_line / np.max(z_line)

        for p in range(len(z_line)):
            center_x, center_y, z_value = x_line[p], y_line[p], z_line[p]
            center_x_int = int(round(center_x))
            center_y_int = int(round(center_y))
            x_start = max(0, center_x_int - weight_radius)
            x_end = min(patch.shape[1], center_x_int + weight_radius + 1)
            y_start = max(0, center_y_int - weight_radius)
            y_end = min(patch.shape[0], center_y_int + weight_radius + 1)
            if x_end <= x_start or y_end <= y_start:
                continue  # Skip invalid regions
            x_indices = np.arange(x_start, x_end)
            y_indices = np.arange(y_start, y_end)
            x_mesh, y_mesh = np.meshgrid(x_indices, y_indices, indexing='xy')
            distances = np.hypot(x_mesh - center_x, y_mesh - center_y)
            gaussian_mask = z_value * np.exp(-distances ** 2 / (2 * (weight_radius / 3) ** 2))
            gaussian_mask[distances > weight_radius] = 0
            weights[y_start:y_end, x_start:x_end] += gaussian_mask

    # Replace centers and bounds with the valid ones
    centers = valid_centers
    bounds = valid_bounds

    if len(centers) == 0:
        logging.error("No valid dislocations found after preprocessing.")
        return None, None, None, None, None  # Maintain original return structure

    # Prepare the data for fitting
    mask = weights > 0.05
    masked_patch = patch[mask]
    if masked_patch.size == 0:
        logging.error("No data to fit after masking.")
        return None, None, None, None, None  # Maintain original return structure

    min_val = np.min(masked_patch)
    max_val = np.max(masked_patch)
    if max_val - min_val == 0:
        logging.error("Zero variance in masked patch.")
        return None, None, None, None, None  # Maintain original return structure

    patch_normalized = (patch - min_val) / (max_val - min_val)
    patch_normalized = patch_normalized - 0.5
    weights_normalized = min_max_normalize(weights)

    # Create x and y coordinates for fit
    sides = []
    for i in range(len(bounds)):
        sides.append(max(bounds[i][2] - bounds[i][0], bounds[i][3] - bounds[i][1]))
    average_side = np.mean(sides) if sides else max(patch.shape)
    pixel_size = 3 / average_side

    # Center the patch around 0,0
    x = (np.arange(patch.shape[1]) - patch.shape[1] // 2) * pixel_size
    y = (np.arange(patch.shape[0]) - patch.shape[0] // 2) * pixel_size

    # Create meshgrid
    x_grid_full, y_grid_full = np.meshgrid(x, y, indexing='xy')

    # Flatten the meshgrid arrays and stack them
    xy = np.vstack((x_grid_full.ravel(), y_grid_full.ravel()))
    z = patch_normalized.ravel()
    weights_flat = weights_normalized.ravel()

    # Replace zeros in weights with a small positive value
    epsilon = 1e-10  # Small positive value
    weights_flat = np.where(weights_flat == 0, epsilon, weights_flat)

    # Initial guesses and bounds
    p0 = []
    bounds_lower = []
    bounds_upper = []
    n_params_per_dislocation = 12

    for i in range(len(centers)):
        x_center, y_center = centers[i]
        translation_x_guess = (peak_x_list[i] - patch.shape[1] // 2) * pixel_size
        translation_y_guess = (peak_y_list[i] - patch.shape[0] // 2) * pixel_size
        stretch_x_guess = 1
        stretch_y_guess = 1
        angle_x_guess = 0
        angle_y_guess = 0
        angle_z_guess = best_angle[i]
        amplitude_guess = 1
        height_factor_guess = 0
        baseline_guess = 0
        shear_x_guess = 0
        shear_y_guess = 0

        p0_i = [
            translation_x_guess, translation_y_guess,
            stretch_x_guess, stretch_y_guess,
            angle_x_guess, angle_y_guess, angle_z_guess,
            amplitude_guess, height_factor_guess, baseline_guess,
            shear_x_guess, shear_y_guess
        ]

        # Adjust bounds based on the initial guesses
        translation_bound_x = (bounds[i][2] - bounds[i][0]) / 3 * pixel_size  # Adjusted to be in the same units
        translation_bound_y = (bounds[i][3] - bounds[i][1]) / 3 * pixel_size
        translation_bounds_x = (
            translation_x_guess - translation_bound_x,
            translation_x_guess + translation_bound_x
        )
        translation_bounds_y = (
            translation_y_guess - translation_bound_y,
            translation_y_guess + translation_bound_y
        )
        stretch_bounds = (0.5, 1.5)
        angle_bounds = (-np.pi, np.pi)
        amplitude_bounds = (0.1, 5)
        height_factor_bounds = (-0.6, 0.6)
        baseline_bounds = (-5, 5)
        shear_bounds = (-5, 5)

        lower_bounds_i = [
            translation_bounds_x[0], translation_bounds_y[0],
            stretch_bounds[0], stretch_bounds[0],
            angle_bounds[0], angle_bounds[0], angle_bounds[0],
            amplitude_bounds[0], height_factor_bounds[0],
            baseline_bounds[0], shear_bounds[0], shear_bounds[0]
        ]
        upper_bounds_i = [
            translation_bounds_x[1], translation_bounds_y[1],
            stretch_bounds[1], stretch_bounds[1],
            angle_bounds[1], angle_bounds[1], angle_bounds[1],
            amplitude_bounds[1], height_factor_bounds[1],
            baseline_bounds[1], shear_bounds[1], shear_bounds[1]
        ]

        # Ensure initial guesses are within bounds
        for idx in range(len(p0_i)):
            if p0_i[idx] < lower_bounds_i[idx]:
                p0_i[idx] = lower_bounds_i[idx] + 1e-5  # Small offset to be within bounds
            elif p0_i[idx] > upper_bounds_i[idx]:
                p0_i[idx] = upper_bounds_i[idx] - 1e-5  # Small offset to be within bounds

        p0.extend(p0_i)
        bounds_lower.extend(lower_bounds_i)
        bounds_upper.extend(upper_bounds_i)

        # Store for plotting (optional, can be removed if not needed)
        translation_x_guesses.append(translation_x_guess)
        translation_y_guesses.append(translation_y_guess)
        translation_bounds_x_list.append(translation_bounds_x)
        translation_bounds_y_list.append(translation_bounds_y)

    # Convert to numpy arrays for optimization
    p0 = np.array(p0)
    bounds_lower = np.array(bounds_lower)
    bounds_upper = np.array(bounds_upper)
    param_bounds = list(zip(bounds_lower, bounds_upper))

    if not fit:
        # If fitting is disabled, return initial guesses and relevant parameters
        popt = p0

        # Extract the translation parameters and convert to pixel units
        centers_of_fits_full_image_coordinates = []
        for i in range(len(centers)):
            translation_x = popt[i * n_params_per_dislocation]
            translation_y = popt[i * n_params_per_dislocation + 1]
            translation_x = int((translation_x / pixel_size) + patch_bounds[0] + patch.shape[1] // 2)
            translation_y = int((translation_y / pixel_size) + patch_bounds[1] + patch.shape[0] // 2)
            centers_of_fits_full_image_coordinates.append((translation_y, translation_x))

        return popt, None, xy, centers_of_fits_full_image_coordinates, pixel_size

    # Define the objective function for Differential Evolution
    def objective_no_mask(params, xy, z, weights):
        try:
            model_z = combined_model(xy, *params)
            if np.any(np.isnan(model_z)) or np.any(np.isinf(model_z)):
                logging.warning("Model produced NaN or Inf values.")
                return 1e20  # Penalize invalid solutions
            residuals = (model_z - z)
            cost = np.sum((residuals * weights) ** 2)
            if np.isnan(cost):
                logging.warning("Cost function returned NaN.")
                return 1e20
            return cost
        except Exception as e:
            logging.error(f"Exception in objective function: {e}")
            return 1e20  # Penalize any exceptions

    # Run Differential Evolution with the robust objective function
    try:
        result = differential_evolution(
            objective_no_mask,
            param_bounds,
            args=(xy, z, weights_flat),
            strategy='best1bin',
            maxiter=1000,
            popsize=15,
            tol=1e-6,
            mutation=(0.5, 1),
            recombination=0.7,
            seed=42,
            disp=True, 
            #workers=-1  # Uncomment this line to enable parallel processing
        )
    except Exception as e:
        logging.error(f"Differential Evolution failed: {e}")
        return None, None, None, None, None  # Maintain original return structure

    popt = result.x

    # Check if the optimizer succeeded
    if result.success:
        logging.info(f"Optimization succeeded with cost {result.fun}")
    else:
        logging.warning(f"Optimization failed: {result.message}")

    # Visualize results
    fitted_z = combined_model(xy, *popt)
    
    # Check for NaNs in fitted_z
    if np.any(np.isnan(fitted_z)) or np.any(np.isinf(fitted_z)):
        logging.error("Fitted model contains NaN or infinite values.")
        return None, None, None, None, None  # Maintain original return structure

    fitted_patch = fitted_z.reshape(patch.shape)
    
    residual_patch = patch_normalized - fitted_patch

    plot = False  # Set to True to enable plotting
    if plot:
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        axes[0].imshow(patch_normalized, cmap='gray')

        # Plot the translation initial guess and the final guess and the translation fit_bounds
        for i in range(len(translation_x_guesses)):
            # Initial translation guess
            tx_guess = translation_x_guesses[i]
            ty_guess = translation_y_guesses[i]

            # Convert the translation guess back to image coordinates
            x_guess = tx_guess / pixel_size + patch.shape[1] // 2
            y_guess = ty_guess / pixel_size + patch.shape[0] // 2

            axes[0].plot(x_guess, y_guess, 'ro', label='Initial Translation Guess' if i == 0 else '')

            # Translation bounds
            tx_bounds = translation_bounds_x_list[i]
            ty_bounds = translation_bounds_y_list[i]

            # Convert bounds back to image coordinates
            x_min_bound = tx_bounds[0] / pixel_size + patch.shape[1] // 2
            x_max_bound = tx_bounds[1] / pixel_size + patch.shape[1] // 2
            y_min_bound = ty_bounds[0] / pixel_size + patch.shape[0] // 2
            y_max_bound = ty_bounds[1] / pixel_size + patch.shape[0] // 2

            # Plot horizontal lines for y bounds
            axes[0].hlines(
                y=[y_min_bound, y_max_bound],
                xmin=x_min_bound,
                xmax=x_max_bound,
                colors='r',
                linestyles='--'
            )
            # Plot vertical lines for x bounds
            axes[0].vlines(
                x=[x_min_bound, x_max_bound],
                ymin=y_min_bound,
                ymax=y_max_bound,
                colors='r',
                linestyles='--'
            )

        axes[0].set_title('Original Patch')
        axes[0].legend()
        axes[1].imshow(fitted_patch, cmap='gray')
        axes[1].set_title('Fitted Model')
        axes[2].imshow(residual_patch, cmap='gray')
        axes[2].set_title('Residuals')
        plt.show()

    # Extract the translation parameters and convert to pixel units
    centers_of_fits_full_image_coordinates = []
    for i in range(len(centers)):
        translation_x = popt[i * n_params_per_dislocation]
        translation_y = popt[i * n_params_per_dislocation + 1]
        translation_x = int((translation_x / pixel_size) + patch_bounds[0] + patch.shape[1] // 2)
        translation_y = int((translation_y / pixel_size) + patch_bounds[1] + patch.shape[0] // 2)
        centers_of_fits_full_image_coordinates.append((translation_y, translation_x))

    return popt, None, xy, centers_of_fits_full_image_coordinates, pixel_size


def iterative_dilate_erode(smoothing_mask, iterations):
    """
    Apply grey_dilation followed by grey_erosion to fill in gaps.

    Parameters:
    -----------
    smoothing_mask: np.ndarray
        The input mask to be processed.
    iterations: int
        The number of dilation and erosion iterations to apply.

    Returns:
    --------
    processed_mask: np.ndarray
        The resulting mask after dilation followed by erosion.
    """
    # Create a circular structuring element
    structuring_element = generate_binary_structure(2, 1)  # Circular structure for 2D

    # Apply dilation multiple times using 'reflect' mode to handle edges better
    dilated_mask = smoothing_mask.copy()
    for _ in range(iterations):
        dilated_mask = grey_dilation(dilated_mask, footprint=structuring_element, mode='reflect')
    
    # Apply erosion multiple times using 'reflect' mode to handle edges better
    eroded_mask = dilated_mask.copy()
    for _ in range(iterations - 1):
        eroded_mask = grey_erosion(dilated_mask, footprint=structuring_element, mode='reflect')

    return eroded_mask

def smooth_and_add_noise(image, z, x_center, y_center, x_min, y_min, x_max, y_max, gradient_threshold=0.01, laplacian_threshold=0.01,
                         smoothing_sigma=15, boundary_blur_sigma=0.5, noise_sigma_estimation=0.3, noise_type='gaussian', 
                         structuring_element_size=5, dilation_iterations=3, grey_dilation_iterations=3, beta=1.0, gamma=1.0, min_smoothing_value=0.1, show_plots=False):

    """
    Function to smooth the image based on a model fit, apply edge-aware smoothing, add noise, and visualize the steps.
    It includes the generation of the mask using gradient and Laplacian thresholds.

    Additional Parameter:
    ---------------------
    grey_dilation_iterations: int
        The number of iterations to apply for grey dilation.
    """
    z = z.reshape(image.shape)

    # Mask the dislocation region: flag pixels where the fitted model is flat
    # (small gradient AND small Laplacian), then invert to get the feature itself
    gradient_y, gradient_x = np.gradient(z)
    gradient_magnitude = np.hypot(gradient_x, gradient_y)

    laplacian_z = np.gradient(gradient_x, axis=1) + np.gradient(gradient_y, axis=0)

    gradient_mask = gradient_magnitude < gradient_threshold
    laplacian_mask = np.abs(laplacian_z) < laplacian_threshold

    combined_mask = np.logical_and(gradient_mask, laplacian_mask)

    inverse_mask = np.logical_not(combined_mask)

    # Baseline intensity: average z over the pixels deepest inside the mask
    # (top 10% by distance from the boundary), so edge pixels that still carry
    # some background don't bias the estimate
    distance_from_boundary = distance_transform_edt(inverse_mask)

    masked_area_values = z[inverse_mask]
    distances_in_masked_area = distance_from_boundary[inverse_mask]

    sorted_indices = np.argsort(distances_in_masked_area)[::-1]
    sorted_values = masked_area_values[sorted_indices]
    sorted_distances = distances_in_masked_area[sorted_indices]

    top_10_percent_index = int(len(sorted_values) * 0.1)
    furthest_10_percent_values = sorted_values[:top_10_percent_index]

    average_value_masked_area = np.mean(furthest_10_percent_values)

    # Smoothing weight ~ deviation from baseline, normalised to [0, 1]
    difference = np.abs(z - average_value_masked_area)

    smoothing_mask = (difference - np.min(difference)) / (np.max(difference) - np.min(difference))

    # Keep a floor of min_smoothing_value so every pixel gets at least some smoothing
    smoothing_mask = smoothing_mask * (1 - min_smoothing_value) + min_smoothing_value

    # Grow the smoothing mask so coverage extends slightly past the fitted feature
    structuring_element = np.ones((structuring_element_size, structuring_element_size))
    dilated_smoothing_mask = iterative_dilate_erode(smoothing_mask, iterations=grey_dilation_iterations)

    blurred_image = gaussian_filter(image, sigma=smoothing_sigma)

    # Binary mask of the feature plus immediate surroundings
    dilated_mask = binary_dilation(inverse_mask, iterations=dilation_iterations)

    # Distance transform inside the dilated mask, normalised, then raised to beta:
    # weighting falls off towards the mask edge so the blend feathers out
    # instead of leaving a hard seam
    distance_map = distance_transform_edt(dilated_mask)
    distance_map_normalized = np.zeros_like(distance_map)
    distance_map_normalized[dilated_mask] = distance_map[dilated_mask] / np.max(distance_map[dilated_mask])

    edge_weighting = distance_map_normalized ** beta
    edge_weighting[~dilated_mask] = 0

    adjusted_smoothing_mask = dilated_smoothing_mask * edge_weighting

    # Blend original and blurred images, weighted by the adjusted mask
    smoothed_image = (1 - adjusted_smoothing_mask) * image + adjusted_smoothing_mask * blurred_image

    # Light extra blur over the masked region to hide any remaining blend boundary
    final_smoothed_image = smoothed_image.copy()
    blurred_region = gaussian_filter(final_smoothed_image, sigma=boundary_blur_sigma)
    final_smoothed_image[dilated_mask] = blurred_region[dilated_mask]

    # Estimate the image noise level as the residual after a light blur
    blurred_for_noise_estimation = gaussian_filter(image, sigma=noise_sigma_estimation)
    noise_estimate = image - blurred_for_noise_estimation
    sigma_noise = np.std(noise_estimate)

    # Regenerate matching noise and add it back so the smoothed patch doesn't
    # stand out as unnaturally clean against the rest of the image.
    # Fixed seed keeps the output reproducible between runs.
    np.random.seed(0)
    if noise_type == 'gaussian':
        noise = np.random.normal(loc=0.0, scale=sigma_noise, size=image.shape)
    elif noise_type == 'uniform':
        a = sigma_noise * np.sqrt(3)
        noise = np.random.uniform(low=-a, high=a, size=image.shape)
    else:
        raise ValueError(f"Unknown noise type: {noise_type}")

    noise_mask = dilated_mask  # include immediate surroundings
    final_image_with_noise = final_smoothed_image.copy()
    final_image_with_noise[noise_mask] += noise[noise_mask]

    # Clip back to the original intensity range
    final_image_with_noise = np.clip(final_image_with_noise, np.min(image), np.max(image))

    if show_plots:
        fig, axes = plt.subplots(3, 4, figsize=(24, 18))

        # Original Image with Bounding Box and Center
        axes[0, 0].imshow(image, cmap='gray')
        axes[0, 0].plot(x_center, y_center, 'ro', label='Dislocation Center')  # Plot the center
        axes[0, 0].plot([x_min, x_max], [y_min, y_min], 'y--')  # Top of bounding box
        axes[0, 0].plot([x_min, x_max], [y_max, y_max], 'y--')  # Bottom of bounding box
        axes[0, 0].plot([x_min, x_min], [y_min, y_max], 'y--')  # Left side of bounding box
        axes[0, 0].plot([x_max, x_max], [y_min, y_max], 'y--')  # Right side of bounding box
        axes[0, 0].set_title('Original Image with Bounding Box')
        axes[0, 0].axis('off')
        axes[0, 0].legend()

        # Fitted Model 'z'
        axes[0, 1].imshow(z, cmap='gray')
        axes[0, 1].set_title('Fitted Model (z)')
        axes[0, 1].axis('off')

        # Original Smoothing Mask
        axes[0, 2].imshow(smoothing_mask, cmap='viridis')
        axes[0, 2].set_title('Original Smoothing Mask')
        axes[0, 2].axis('off')

        # Dilated Smoothing Mask
        axes[0, 3].imshow(dilated_smoothing_mask, cmap='viridis')
        axes[0, 3].set_title('Dilated Smoothing Mask (Iterative)')
        axes[0, 3].axis('off')

        # Edge Weighting
        axes[1, 0].imshow(edge_weighting, cmap='viridis')
        axes[1, 0].set_title('Edge Weighting')
        axes[1, 0].axis('off')

        # Adjusted Smoothing Mask
        axes[1, 1].imshow(adjusted_smoothing_mask, cmap='viridis')
        axes[1, 1].set_title('Adjusted Smoothing Mask')
        axes[1, 1].axis('off')

        # Initial Smoothed Image
        axes[1, 2].imshow(smoothed_image, cmap='gray')
        axes[1, 2].set_title('Smoothed Image (Before Final Blur)')
        axes[1, 2].axis('off')

        # Boundary Mask
        axes[1, 3].imshow(dilated_mask, cmap='gray')
        axes[1, 3].set_title('Dilated Mask (Boundary)')
        axes[1, 3].axis('off')

        # Final Smoothed Image (Before Noise)
        axes[2, 0].imshow(final_smoothed_image, cmap='gray')
        axes[2, 0].set_title('Final Smoothed Image (Before Noise)')
        axes[2, 0].axis('off')

        # Noise Added to Smoothed Area
        axes[2, 1].imshow(noise * noise_mask, cmap='gray')
        axes[2, 1].set_title('Noise Added to Smoothed Area')
        axes[2, 1].axis('off')

        # Final Image with Noise
        axes[2, 2].imshow(final_image_with_noise, cmap='gray')
        axes[2, 2].set_title('Final Image with Noise Added Back')
        axes[2, 2].axis('off')

        # Empty plot for alignment
        axes[2, 3].axis('off')

        plt.tight_layout()
        plt.show()

    return final_image_with_noise
