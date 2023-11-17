import cv2

def mouse_click_event(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(f"Clicked at coordinates (x: {x}, y: {y})")

def read_and_display_image(image_path):
    # Read the image using OpenCV
    image = cv2.imread(image_path)

    if image is None:
        print(f"Error: Unable to read the image at '{image_path}'.")
        return

    # Display the image
    cv2.imshow("Image", image)

    # Set the callback function for mouse events
    cv2.setMouseCallback("Image", mouse_click_event)

    # Wait for a key press and close the window when a key is pressed
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    idx = str(12).zfill(6)
    image_path = f"/home/praveenvnktsh/alfred-autonomy/src/manipulation/scripts/images/rgb/{idx}.png"  # Replace with the path to your image
    read_and_display_image(image_path)
