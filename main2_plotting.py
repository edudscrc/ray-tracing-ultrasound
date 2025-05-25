import numpy as np
import matplotlib.pyplot as plt

def load_and_visualize_data(filepath):
    """
    Loads simulation data from a .npy file and visualizes it.
    """
    try:
        # Load the structured array
        loaded_data = np.load(filepath)
        print(f"Successfully loaded data from: {filepath}")
        print(f"Shape of loaded data: {loaded_data.shape}")
        print(f"Data type (dtype) of loaded data: {loaded_data.dtype}")
    except FileNotFoundError:
        print(f"Error: The file '{filepath}' was not found.")
        return
    except Exception as e:
        print(f"An error occurred while loading the file: {e}")
        return

    if loaded_data.size == 0:
        print("The loaded data array is empty. No data to display or plot.")
        return

    # --- Printing some of the results ---
    print("\n--- Sample of Loaded Data (first up to 5 entries) ---")
    for i in range(min(5, len(loaded_data))):
        entry = loaded_data[i]
        print(f"Entry {i}: Emitter Idx: {entry['emitter_idx']}, Receiver Idx: {entry['receiver_idx']}, "
              f"Receiver X: {entry['receiver_x_pos']:.4f}, Min TOF: {entry['min_tof']:.3e} s, "
              f"Alpha (Emitter): {entry['alpha_emitter_deg']:.2f}°, Alpha Idx: {entry['alpha_emitter_idx']}")

    # --- Plotting the collected information ---
    # Extract data for plotting
    # These field names must match what you used when saving the structured array
    try:
        plot_receiver_indices = loaded_data['receiver_idx']
        plot_min_tofs = loaded_data['min_tof']
        plot_alpha_degrees = loaded_data['alpha_emitter_deg']
        # Assuming all entries are from the same emitter for the title, get it from the first entry
        emitter_element_idx_loaded = loaded_data[0]['emitter_idx'] if len(loaded_data) > 0 else 'Unknown'
    except ValueError as e:
        print(f"\nError accessing data fields for plotting. Make sure field names are correct: {e}")
        print(f"Available field names in loaded data: {loaded_data.dtype.names}")
        return


    fig, ax = plt.subplots(figsize=(14, 8))
    scatter_plot = ax.scatter(plot_receiver_indices, plot_min_tofs,
                               c=plot_alpha_degrees, cmap='viridis',
                               s=150, alpha=0.8, edgecolors='k')

    ax.set_xlabel("Hit Element Index (Receiver)")
    ax.set_ylabel("Minimum Time of Flight (s)")
    ax.set_title(f"Minimum TOF to Receiver Elements (Emitter Element: {emitter_element_idx_loaded}) - Loaded Data")

    cbar = fig.colorbar(scatter_plot, ax=ax)
    cbar.set_label("Emitter's Initial Alpha for Ray (degrees)")

    if len(plot_receiver_indices) > 0:
        ax.set_xticks(np.unique(plot_receiver_indices).astype(int))

    ax.grid(True, linestyle=':', alpha=0.7)

    # --- Adding hover annotation (similar to the previous script) ---
    annot = ax.annotate("", xy=(0,0), xytext=(15,15), textcoords="offset points",
                        bbox=dict(boxstyle="round,pad=0.5", fc="yellow", alpha=0.75),
                        arrowprops=dict(arrowstyle="->"))
    annot.set_visible(False)

    # The loaded_data is already in the correct format (structured array)
    # where each element corresponds to a point.
    
    def update_annot(ind):
        pos = scatter_plot.get_offsets()[ind["ind"][0]]
        annot.xy = pos
        
        point_data = loaded_data[ind["ind"][0]] # Direct lookup in the structured array
        
        text = (f"Receiver Idx: {point_data['receiver_idx']}\n"
                f"Min TOF: {point_data['min_tof']:.3e} s\n"
                f"Emitter Alpha: {point_data['alpha_emitter_deg']:.2f}°")
        annot.set_text(text)

    def hover(event):
        vis = annot.get_visible()
        if event.inaxes == ax:
            cont, ind = scatter_plot.contains(event)
            if cont:
                update_annot(ind)
                annot.set_visible(True)
                fig.canvas.draw_idle()
            else:
                if vis:
                    annot.set_visible(False)
                    fig.canvas.draw_idle()

    fig.canvas.mpl_connect("motion_notify_event", hover)
    
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Prompt the user for the filepath or use a default
    default_filename = "simulation_emitter_32_paths.npy" # Change if your default is different
    filepath_to_load = input(f"Enter the path to the .npy file (default: {default_filename}): ")
    if not filepath_to_load:
        filepath_to_load = default_filename
    
    load_and_visualize_data(filepath_to_load)