import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mplcursors
from matplotlib.widgets import Slider
from matplotlib.widgets import Button

#K-means cluster finding algorithm based on an integer for the amount of clusters and a max iterations
def kmeans(data, k, max_iterations=100):
    # Randomly initialize centroids
    centroids = data.sample(n=k).values
    
    for _ in range(max_iterations):
        # Assign each data point to the nearest centroid
        distances = np.sqrt(((data.values - centroids[:, np.newaxis])**2).sum(axis=2))
        labels = np.argmin(distances, axis=0)
        
        # Update centroids by taking the mean of assigned points
        new_centroids = np.array([data.values[labels == i].mean(axis=0) for i in range(k)])
        
        # If centroids have not changed, terminate
        if np.all(centroids == new_centroids):
            break
        
        centroids = new_centroids
    
    return centroids, labels


def write_data_to_csv(data, filename):
    # Write DataFrame to a CSV file
    data.to_csv(filename, index=False)

#make random clusters based on input
def generate_random_data(num_clusters, num_points_per_cluster, scale_factor): 
    # Randomly generate cluster centers
    cluster_centers = np.random.uniform(low=0, high=20, size=(num_clusters, 2))

    # Generate data points around cluster centers
    data_points = []
    for center in cluster_centers:
        x = np.random.normal(loc=center[0], scale=scale_factor, size=num_points_per_cluster)
        y = np.random.normal(loc=center[1], scale=scale_factor, size=num_points_per_cluster)
        data_points.extend(list(zip(x, y)))

    # Create a DataFrame with the generated data
    data = pd.DataFrame(data_points, columns=['x', 'y'])

    return data 

#update the clusters you search for when changing the sliders
def update_clusters(val):
    global k, max_iterations
    
    k = int(slider_k.val)
    max_iterations = int(slider_iterations.val)
    centroids, labels = kmeans(data, k, max_iterations)
    scatter.set_array(labels)
    scatter_centroids.set_offsets(centroids)
    plt.title(f'K-means Clustering (k = {k}, max_iterations = {max_iterations})')
    plt.draw()


#update the random data based on the values of the input
def regenerate_data(event):
    global data, k, max_iterations
    
    num_clusters = int(slider_num_clusters.val)
    print("The number of new clusters should be:", num_clusters)
    
    num_points = int(slider_num_points.val)
    print("The number of new points should be:", num_points)
    
    scale_factor = int(slider_scale_factor.val)
    print("The scale factor should be:", scale_factor)
    
    data = generate_random_data(num_clusters, num_points, scale_factor)
    
    # Update scatter plot and centroids
    scatter.set_offsets(data.values[:, :2])  # Use a 2D NumPy array
    scatter_centroids.set_offsets(np.empty((0, 2)))  # Clear centroids
    plt.title(f'K-means Clustering (k = {k}, max_iterations = {max_iterations})')
    plt.draw()



# Generate random data with 100 points
num_clusters = 4
num_points = 300
scale_factor = 1
data = generate_random_data(num_clusters, num_points, scale_factor)

# Write the data to a CSV file
filename = 'generated_data.csv'
write_data_to_csv(data, filename)

# Read data from CSV file
data = pd.read_csv(filename)

# Perform K-means clustering
k = 10
max_iterations = 100
centroids, labels = kmeans(data, k, max_iterations)

# Create the figure and subplot
fig, ax = plt.subplots()
plt.subplots_adjust(bottom=0.5)

# Create sliders for changing the parameters
ax_slider_k = plt.axes([0.25, 0.4, 0.5, 0.03])
slider_k = Slider(ax_slider_k, 'Num Clusters', 1, 10, valinit=k, valstep=1)

ax_slider_iterations = plt.axes([0.25, 0.35, 0.5, 0.03])
slider_iterations = Slider(ax_slider_iterations, 'Max Iterations', 10, 200, valinit=max_iterations, valstep=10)

ax_slider_num_clusters = plt.axes([0.25, 0.3, 0.5, 0.03])
slider_num_clusters = Slider(ax_slider_num_clusters, 'Num Clusters', 1, 10, valinit=num_clusters, valstep=1)

ax_slider_num_points = plt.axes([0.25, 0.25, 0.5, 0.03])
slider_num_points = Slider(ax_slider_num_points, 'Num Points', 100, 500, valinit=num_points, valstep=10)

ax_slider_scale_factor = plt.axes([0.25, 0.2, 0.5, 0.03])
slider_scale_factor = Slider(ax_slider_scale_factor, 'Scale Factor', 1, 3, valinit=scale_factor, valstep=0.01)

ax_regenerate_button = plt.axes([0.4, 0.1, 0.2, 0.05])
regenerate_button = Button(ax_regenerate_button, 'Regenerate Data')

# Connect the sliders' on_changed events to update_clusters and update_centroids functions
slider_k.on_changed(update_clusters)
slider_iterations.on_changed(update_clusters)

# Connect the button's on_clicked event to regenerate_data function
regenerate_button.on_clicked(regenerate_data)

# Define colormap
cmap = plt.cm.get_cmap('tab10', 10)

# Create scatter plot and centroids
scatter = ax.scatter(data['x'], data['y'], c=labels, cmap=cmap)
scatter_centroids = ax.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='x')
ax.set_xlabel('x')
ax.set_ylabel('y')

# Add interactivity to display distances
cursor = mplcursors.cursor(scatter)


@cursor.connect("add")
def on_add(sel):
    index = sel.target.index
    point = data.iloc[index]
    
    k = int(slider_k.val)
    max_iterations = int(slider_iterations.val)
    centroids, labels = kmeans(data, k, max_iterations)
    
    distances = np.sqrt(((point.values - centroids)**2).sum(axis=1))
    min_distance_index = np.argmin(distances)

    distances_text = "\n".join(f"Distance to Centroid {i+1}: {d:.2f}" if i != min_distance_index
                              else f"Closest Centroid {i+1}: {d:.2f}"
                              for i, d in enumerate(distances))

    sel.annotation.set(text=distances_text, position=(10, 10), anncoords="axes points", alpha=0.7)

plt.show()
