import cv2
import numpy as np
import os
import numpy as np
directory = os.path.dirname(__file__)

print(directory)
# Set the path to the folder containing the screenshots
folder_path =directory + "/video_frames/reshelving/pick/"
files = os.listdir(folder_path)
image_files = [f for f in files if f.endswith(".png") and not(f.startswith("template"))]
image_files.sort()

for i, image_file in enumerate(image_files):
    image_path = os.path.join(folder_path, image_file)
    # Load the image
    image = cv2.imread(image_path)

    # Create a blank mask
    mask = np.zeros_like(image)

    # List to store points
    points = []

    # Mouse callback function
    def draw_polygon(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append((x, y))
        elif event == cv2.EVENT_RBUTTONDOWN and len(points) > 1:
            cv2.polylines(mask, [np.array(points)], True, (255, 255, 255), 2)
            points.clear()

    # Set mouse callback
    cv2.namedWindow('image')
    cv2.setMouseCallback('image', draw_polygon)
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    while True:
        cv2.imshow('image', cv2.addWeighted(image, 0.8, mask, 0.2, 0))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    points_array = np.array(points, dtype=np.int32).reshape((-1, 1, 2))
    # Fill the mask
    cv2.fillPoly(mask, [points_array], (255, 255, 255))

    # Apply the mask to the image
    image[mask==0] = 0

    # Save the image
    cv2.imwrite(folder_path+'/cropped/'+ image_file, image)

    cv2.destroyAllWindows()