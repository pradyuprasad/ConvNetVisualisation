import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns # type: ignore
from pathlib import Path
from analyse_projections import load_and_structure_data, analyze_filter_activity
import json

def visualize_top_filters(data_dir: Path, json_path: Path, output_dir: Path):
   # Create output directory if it doesn't exist
   output_dir.mkdir(exist_ok=True)

   # Load and analyze data
   with open(json_path) as f:
       json_data = json.load(f)
   filter_activations, class_names, class_indices = load_and_structure_data(json_data)
   results = analyze_filter_activity(filter_activations, class_names, class_indices)

   # Get top filter for each class
   for class_name in class_names:
       top_filter = results['top_filters_per_class'][class_name][0]

       # Load first image for this class
       npz_files = list(data_dir.glob(f"{class_name}_*.npz"))
       data = np.load(npz_files[0])
       filter_viz = data['visualizations'][top_filter]
       intensity = np.mean(filter_viz, axis=2)

       # Create single heatmap
       plt.figure(figsize=(10, 8))
       sns.heatmap(intensity, cmap='viridis', cbar=True)
       plt.title(f'Most Active Filter for {class_name}\nFilter {top_filter}')

       # Save figure
       save_path = output_dir / f'{class_name}_top_filter_{top_filter}.png'
       plt.savefig(save_path)
       plt.show()
       plt.close()

if __name__ == "__main__":
   data_dir = Path("stl10_deconv_visualizations")
   json_path = Path("filter_metrics.json")
   output_dir = Path("filter_heatmaps")
   visualize_top_filters(data_dir, json_path, output_dir)
