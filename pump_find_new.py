import cv2
import numpy as np
import networkx as nx
from io import BytesIO

def generate_pump_image(state_name):
    # Load the image
    image_path = f'C:\\Amitesh\\cc\\syntax_error_backend\\{state_name}.jpg'
    image = cv2.imread(image_path)
    
    if image is None:
        raise FileNotFoundError(f'Image for state "{state_name}" not found.')

    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Define color ranges for petrol pumps, population density, and roads
    lower_pump = np.array([0, 200, 0])
    upper_pump = np.array([0, 255, 0])
    mask_pumps = cv2.inRange(image, lower_pump, upper_pump)

    lower_density = 30
    upper_density = 100
    mask_density = cv2.inRange(gray, lower_density, upper_density)

    lower_road = np.array([50, 0, 0])
    upper_road = np.array([255, 0, 0])
    mask_roads = cv2.inRange(image, lower_road, upper_road)

    # Find contours for pumps, density regions, and roads
    contours_pumps, _ = cv2.findContours(mask_pumps, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours_density, _ = cv2.findContours(mask_density, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours_roads, _ = cv2.findContours(mask_roads, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create a graph to represent the road network
    road_graph = nx.Graph()

    for contour in contours_roads:
        for point in contour:
            road_graph.add_node(tuple(point[0]), pos=tuple(point[0]))

    for contour in contours_roads:
        for i in range(len(contour) - 1):
            p1 = tuple(contour[i][0])
            p2 = tuple(contour[i + 1][0])
            distance = np.linalg.norm(np.array(p1) - np.array(p2))
            road_graph.add_edge(p1, p2, weight=distance)

    contours_pumps = list(contours_pumps)
    new_pump_locations = []
    safety = 0
    penalty_radius = 100

    # Main loop to place pumps until no dark zones remain
    while True:
        score_map = np.zeros(gray.shape)

        for contour in contours_density:
            if cv2.contourArea(contour) > 1000:
                cv2.drawContours(score_map, [contour], -1, 255, thickness=cv2.FILLED)

        for contour in contours_pumps:
            for pump_point in contour:
                pump_coords = tuple(pump_point[0])
                cv2.circle(score_map, pump_coords, penalty_radius, 0, thickness=-1)

        dark_zone_found = False
        for contour in contours_density:
            if cv2.contourArea(contour) > 1000:
                dark_zone_found = True
                _, max_val, _, max_loc = cv2.minMaxLoc(score_map)
                if max_val == 255:
                    new_pump_locations.append(max_loc)
                    cv2.circle(image, max_loc, 10, (0, 255, 0), thickness=-1)
                    contours_pumps.append(np.array([[max_loc]]))

        if not dark_zone_found:
            break

        safety += 1
        if safety > 100:
            break

    for loc in new_pump_locations:
        cv2.circle(image, loc, 10, (255, 0, 0), thickness=-1)

    # Save the resulting image to a buffer
    buffer = BytesIO()
    _, img_encoded = cv2.imencode('.png', image)
    buffer.write(img_encoded)
    buffer.seek(0)

    return buffer