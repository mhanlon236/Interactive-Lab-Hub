import math

def calculate_angle(landmark1, landmark2, landmark3):
    x1, y1 = landmark1
    x2, y2 = landmark2
    x3, y3 = landmark3

    # Calculate vectors
    vector1 = (x1 - x2, y1 - y2)
    vector2 = (x3 - x2, y3 - y2)

    # Calculate dot product
    dot_product = vector1[0] * vector2[0] + vector1[1] * vector2[1]

    # Calculate magnitudes
    magnitude1 = math.sqrt(vector1[0]**2 + vector1[1]**2)
    magnitude2 = math.sqrt(vector2[0]**2 + vector2[1]**2)

    # Calculate angle in radians
    angle_rad = math.acos(dot_product / (magnitude1 * magnitude2))

    # Convert angle to degrees
    angle_deg = math.degrees(angle_rad)

    return angle_deg

# Example usage:
landmark1 = (x1, y1)  # Replace with actual landmark coordinates
landmark2 = (x2, y2)  # Replace with actual landmark coordinates
landmark3 = (x3, y3)  # Replace with actual landmark coordinates

angle = calculate_angle(landmark1, landmark2, landmark3)
print(f"Angle: {angle} degrees")
