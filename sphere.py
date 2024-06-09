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

# Function to compute ray directions for all pixels in the image
def get_ray_directions(width, height):
    u = torch.linspace(0, 1, width, device=device)
    v = torch.linspace(0, 1, height, device=device)
    u, v = torch.meshgrid(u, v)
    directions = lower_left_corner + u.unsqueeze(-1) * horizontal + v.unsqueeze(-1) * vertical - camera_origin
    return directions

# Function to check ray-sphere intersection
def ray_sphere_intersection(origin, direction, center, radius):
    oc = origin - center
    a = torch.sum(direction ** 2, dim=-1)
    b = 2.0 * torch.sum(oc * direction, dim=-1)
    c = torch.sum(oc ** 2, dim=-1) - radius ** 2
    discriminant = b ** 2 - 4 * a * c
    hit = (discriminant >= 0)
    t = (-b - torch.sqrt(discriminant)) / (2.0 * a)
    t[discriminant < 0] = float('inf')
    return hit, t

# Render the scene
def render_scene(width, height):
    directions = get_ray_directions(width, height)
    directions = F.normalize(directions, p=2, dim=-1)
    hit, t = ray_sphere_intersection(camera_origin, directions, sphere_center, sphere_radius)
    pixels = torch.zeros((height, width, 3), device=device)
    hit_pixels = hit.nonzero()
    if hit_pixels.numel() > 0:
        hit_directions = directions[hit_pixels[:, 0], hit_pixels[:, 1]]
        hit_t = t[hit_pixels[:, 0], hit_pixels[:, 1]]
        hit_points = camera_origin + hit_t.unsqueeze(-1) * hit_directions
        hit_normals = F.normalize(hit_points - sphere_center, p=2, dim=-1)
        hit_colors = 0.5 * (hit_normals + 1.0) * sphere_color
        pixels[hit_pixels[:, 0], hit_pixels[:, 1]] = hit_colors
    return pixels.cpu().numpy()

# Main function
if __name__ == "__main__":
    image = render_scene(width, height)
    plt.imshow(image)
    plt.axis('off')
    plt.show()
