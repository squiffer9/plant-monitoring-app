import cv2


def test_camera(device_id=0):
    cap = cv2.VideoCapture(device_id)
    if not cap.isOpened():
        print(f"Error: Could not open camera with device ID {device_id}.")
        return

    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
    else:
        cv2.imshow("Test Frame", frame)
        cv2.waitKey(0)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    for device_id in range(5):  # 試行可能なデバイスIDの範囲
        print(f"Trying device ID {device_id}")
        test_camera(device_id)
