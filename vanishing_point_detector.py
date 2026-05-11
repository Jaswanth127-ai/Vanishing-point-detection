# ============================================================
# VANISHING POINT DETECTION USING RANSAC + HOUGH TRANSFORM
# ============================================================

# Import required libraries
import cv2
import numpy as np
import random
import matplotlib.pyplot as plt

# ============================================================
# 1. LOAD INPUT IMAGE
# ============================================================

# Path to the input image
image_path = r"C:\Users\Andhavarapu Jahnavi\Desktop\tooth vs non tooth\Estimate_vanishing_points_data\pexels-photo-10622719.jpeg"

# Read image
img = cv2.imread(image_path)

# Check if image exists
if img is None:
    raise FileNotFoundError("Image not found!")

# Keep a copy of original image
original = img.copy()

# ============================================================
# 2. IMAGE PREPROCESSING
# ============================================================

# Convert image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Apply Gaussian Blur to reduce noise
gray = cv2.GaussianBlur(gray, (5, 5), 0)

# Detect edges using Canny Edge Detector
edges = cv2.Canny(gray, 50, 150)

# ============================================================
# 3. DETECT LINE SEGMENTS USING HOUGH TRANSFORM
# ============================================================

lines = cv2.HoughLinesP(
    edges,                 # Input edge image
    rho=1,                 # Distance resolution in pixels
    theta=np.pi / 180,     # Angle resolution in radians
    threshold=80,          # Minimum votes needed
    minLineLength=60,      # Minimum line length
    maxLineGap=20          # Maximum allowed gap
)

# Store valid line segments
line_segments = []

if lines is not None:

    for line in lines:

        # Extract coordinates
        x1, y1, x2, y2 = line[0]

        # Compute slope
        slope = (y2 - y1) / (x2 - x1 + 1e-6)

        # Ignore nearly horizontal lines
        if abs(slope) < 0.3:
            continue

        line_segments.append((x1, y1, x2, y2))

# ============================================================
# 4. FUNCTION TO FIND INTERSECTION OF TWO LINES
# ============================================================

def intersection(line1, line2):

    x1, y1, x2, y2 = line1
    x3, y3, x4, y4 = line2

    # Compute denominator
    denom = (x1 - x2)*(y3 - y4) - (y1 - y2)*(x3 - x4)

    # Parallel lines check
    if abs(denom) < 1e-6:
        return None

    # Compute intersection point
    px = ((x1*y2 - y1*x2)*(x3 - x4) -
          (x1 - x2)*(x3*y4 - y3*x4)) / denom

    py = ((x1*y2 - y1*x2)*(y3 - y4) -
          (y1 - y2)*(x3*y4 - y3*x4)) / denom

    return np.array([px, py])

# ============================================================
# 5. FUNCTION TO COMPUTE POINT-TO-LINE DISTANCE
# ============================================================

def point_line_distance(point, line):

    x0, y0 = point
    x1, y1, x2, y2 = line

    numerator = abs(
        (y2 - y1)*x0 -
        (x2 - x1)*y0 +
        x2*y1 -
        y2*x1
    )

    denominator = np.sqrt(
        (y2 - y1)**2 +
        (x2 - x1)**2
    )

    return numerator / denominator

# ============================================================
# 6. RANSAC-BASED VANISHING POINT ESTIMATION
# ============================================================

best_vp = None
best_inliers = []

# Number of RANSAC iterations
iterations = 1000

# Distance threshold
threshold = 10

# Ensure enough lines exist
if len(line_segments) >= 2:

    for _ in range(iterations):

        # Randomly select two lines
        l1, l2 = random.sample(line_segments, 2)

        # Compute candidate vanishing point
        vp_candidate = intersection(l1, l2)

        if vp_candidate is None:
            continue

        inliers = []

        # Check how many lines agree with this VP
        for line in line_segments:

            dist = point_line_distance(vp_candidate, line)

            if dist < threshold:
                inliers.append(line)

        # Keep best solution
        if len(inliers) > len(best_inliers):
            best_inliers = inliers
            best_vp = vp_candidate

# ============================================================
# 7. DRAW RESULTS
# ============================================================

# Copy images for visualization
line_img = original.copy()
result_img = original.copy()

# Draw RANSAC inlier lines
for x1, y1, x2, y2 in best_inliers:
    cv2.line(line_img, (x1, y1), (x2, y2), (0, 255, 0), 3)

# Draw vanishing point
if best_vp is not None:

    vp = tuple(np.int32(best_vp))

    cv2.circle(result_img, vp, 10, (0, 0, 255), -1)

# ============================================================
# 8. DISPLAY OUTPUT
# ============================================================

plt.figure(figsize=(14, 6))

# Display filtered lines
plt.subplot(1, 2, 1)
plt.title("Filtered Lines (RANSAC Inliers)")
plt.imshow(cv2.cvtColor(line_img, cv2.COLOR_BGR2RGB))
plt.axis("off")

# Display vanishing point
plt.subplot(1, 2, 2)
plt.title("Estimated Vanishing Point")
plt.imshow(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB))
plt.axis("off")

plt.show()