import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Image dimensions
width = 800
height = 600

# Sphere parameters
sphere_center = torch.tensor([0.0, 0.0, -5.0], device=device)  # Sphere center
sphere_radius = 1.0  # Sphere radius
sphere_color = torch.tensor([1.0, 0.0, 0.0], device=device)  # Red color

# Camera parameters
camera_origin = torch.tensor([0.0, 0.0, 0.0], device=device)  # Camera position
viewport_height = 2.0
viewport_width = (viewport_height * width) / height
focal_length = 1.0

# Viewport basis vectors
horizontal = torch.tensor([viewport_width, 0.0, 0.0], device=device)
vertical = torch.tensor([0.0, viewport_height, 0.0], device=device)
lower_left_corner = camera_origin - horizontal / 2 - vertical / 2 - torch.tensor([0.0, 0.0, focal_length], device=device)

# Function to compute ray direction
def get_ray_direction(u, v):
    return lower_left_corner + u * horizontal + v * vertical - camera_origin

# Function to check ray-sphere intersection
def ray_sphere_intersection(origin, direction, center, radius):
    oc = origin - center
    a = torch.dot(direction, direction)
    b = 2.0 * torch.dot(oc, direction)
    c = torch.dot(oc, oc) - radius ** 2
    discriminant = b ** 2 - 4 * a * c
    if discriminant < 0:
        return False, None
    t = (-b - torch.sqrt(discriminant)) / (2.0 * a)
    return True, t

# Render the scene
def render_scene(width, height):
    pixels = torch.zeros((height, width, 3), device=device)
    for j in range(height):
        for i in range(width):
            u = i / (width - 1)
            v = j / (height - 1)
            direction = get_ray_direction(u, v)
            # Normalize the direction using F.normalize
            direction = F.normalize(direction, p=2, dim=0)
            hit, t = ray_sphere_intersection(camera_origin, direction, sphere_center, sphere_radius)
            if hit:
                point = camera_origin + t * direction
                normal = F.normalize(point - sphere_center, p=2, dim=0)
                color = 0.5 * (normal + 1.0) * sphere_color
                pixels[j, i] = color
    return pixels.cpu().numpy()

# Main function
if __name__ == "__main__":
    image = render_scene(width, height)
    plt.imshow(image)
    plt.axis('off')
    plt.show()
