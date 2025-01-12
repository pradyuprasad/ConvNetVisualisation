import numpy as np
import json
import glob
import os
from typing import Any, TypedDict, List, Dict, Tuple

from collections import defaultdict
import matplotlib.pyplot as plt



class Metrics(TypedDict):
    non_zero_counts: List[float]
    means: List[float]
    maxes: List[float]
    variances: List[float]

class ImageResult(TypedDict):
    class_name: str
    class_id: int
    image_id: int
    metrics: Metrics

# Get all npz files
def make_summary_stats_of_projections() -> None:
    files = glob.glob('stl10_deconv_visualizations/*.npz')
    assert len(files) == 320, f"Expected 320 files, got {len(files)}"

    results: dict[str, ImageResult] = {}

    for file in files:
        # Extract meaningful parts from filename
        basename = os.path.basename(file)
        print(f" analysing {basename}")
        class_name = basename.split('_class_')[0]
        class_id = int(basename.split('_class_')[1].split('_')[0])
        image_id = int(basename.split('_image_')[1].split('.')[0])

        # Load and process file
        with np.load(file, allow_pickle=True) as data:
            arr = data['visualizations']
            assert arr.shape == (256, 96, 96, 3), f"Wrong shape in {file}: {arr.shape}"

            # Sum across RGB
            arr = np.sum(arr, axis=3)
            assert arr.shape == (256, 96, 96)

            # Calculate metrics
            non_zero_count = np.count_nonzero(arr, axis=(1,2))
            assert non_zero_count.shape == (256,)

            sum_filter = np.sum(arr, axis=(1,2))
            assert sum_filter.shape == (256,)

            mean_vals = np.nan_to_num(sum_filter/non_zero_count)
            assert mean_vals.shape == (256,)

            max_vals = np.max(arr, axis=(1,2))
            assert max_vals.shape == (256,)

            var_vals = np.var(arr, axis=(1,2))
            assert var_vals.shape == (256,)

            # Create meaningful key
            key = f"{class_name}_class{class_id}_img{image_id}"

            # Store results
            results[key] = {
                'class_name': class_name,
                'class_id': class_id,
                'image_id': image_id,
                'metrics': {
                    'non_zero_counts': non_zero_count.tolist(),
                    'means': mean_vals.tolist(),
                    'maxes': max_vals.tolist(),
                    'variances': var_vals.tolist()
                }
            }

    # Save to JSON
    with open('filter_metrics.json', 'w') as f:
        json.dump(results, f)

    # Final sanity checks
    assert len(results) == 320, f"Processed {len(results)} images instead of 320"
    first_key = next(iter(results))
    assert len(results[first_key]['metrics']['non_zero_counts']) == 256, "Wrong number of filters"




def load_and_structure_data(json_data: dict) -> Tuple[np.ndarray, List[str], Dict[str, List[int]]]:
    """
    Converts JSON data into efficiently structured numpy arrays.
    Returns:
        - filter_activations: array of shape (n_images, n_filters, n_metrics)
        - class_names: list of unique class names
        - class_indices: dictionary mapping class names to their image indices
    """
    n_images = len(json_data)
    n_filters = 256
    n_metrics = 4  # non_zero_counts, means, maxes, variances

    # Pre-allocate array
    filter_activations = np.zeros((n_images, n_filters, n_metrics))

    # Track classes
    class_indices = defaultdict(list)

    # Fill data
    for idx, (key, data) in enumerate(json_data.items()):
        metrics = data['metrics']
        filter_activations[idx, :, 0] = metrics['non_zero_counts']
        filter_activations[idx, :, 1] = metrics['means']
        filter_activations[idx, :, 2] = metrics['maxes']
        filter_activations[idx, :, 3] = metrics['variances']

        class_indices[data['class_name']].append(idx)

    class_names = sorted(class_indices.keys())

    return filter_activations, class_names, class_indices

def analyze_filter_activity(filter_activations: np.ndarray,
                          class_names: List[str],
                          class_indices: Dict[str, List[int]],
                          activation_threshold: float = 0.1) -> Dict[str, Any]:
    """
    Analyze filter activation patterns across the dataset.
    """
    # Overall filter activity (normalized by number of images)
    overall_activity = filter_activations[:, :, 0].mean(axis=0)  # shape: (n_filters,)

    # Per-class filter activity
    class_activities = {}
    for class_name in class_names:
        indices = class_indices[class_name]
        class_activities[class_name] = filter_activations[indices, :, 0].mean(axis=0)

    # Find dead filters (filters with very low overall activation)
    dead_filters = np.where(overall_activity < activation_threshold)[0]

    # Find top active filters per class
    top_filters_per_class = {
        class_name: np.argsort(activity)[-10:][::-1]
        for class_name, activity in class_activities.items()
    }

    # Overall top active filters
    top_filters_overall = np.argsort(overall_activity)[-10:][::-1]

    return {
        'overall_activity': overall_activity,
        'class_activities': class_activities,
        'dead_filters': dead_filters,
        'top_filters_per_class': top_filters_per_class,
        'top_filters_overall': top_filters_overall
    }

def visualize_results(results: Dict[str, Any], class_names: List[str]) -> None:
    """
    Create visualizations of the analysis results.
    """
    # Plot overall filter activity
    plt.figure(figsize=(15, 5))
    plt.plot(results['overall_activity'])
    plt.title('Overall Filter Activity')
    plt.xlabel('Filter Index')
    plt.ylabel('Average Activation')
    plt.show()

    # Plot per-class top filters
    plt.figure(figsize=(15, 8))
    for i, class_name in enumerate(class_names):
        activities = results['class_activities'][class_name]
        plt.plot(activities, label=class_name, alpha=0.7)
    plt.title('Filter Activity by Class')
    plt.xlabel('Filter Index')
    plt.ylabel('Average Activation')
    plt.legend()
    plt.show()

# Main execution
def main():
    if not os.path.isfile('filter_metrics.json'):
        make_summary_stats_of_projections()

    with open('filter_metrics.json', 'r') as f:
        data = json.load(f)

    # Structure the data
    filter_activations, class_names, class_indices = load_and_structure_data(data)

    # Analyze
    results = analyze_filter_activity(filter_activations, class_names, class_indices)

    # Print some insights
    print("Dead filters:", len(results['dead_filters']))
    print("\nTop 10 most active filters overall:", results['top_filters_overall'])

    print("\nTop 5 filters per class:")
    for class_name in class_names:
        print(f"{class_name}: {results['top_filters_per_class'][class_name][:5]}")

    # Visualize
    visualize_results(results, class_names)

if __name__ == "__main__":
    main()
