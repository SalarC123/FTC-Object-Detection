import cv2

scale_factor = 3

def rescale_frame(scale_factor, frame):
    width = int(frame.shape[1] / scale_factor)
    height = int(frame.shape[0] / scale_factor)
    dim = (width, height)
    return cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)


cap = cv2.VideoCapture("vid.mp4")

if cap.isOpened():
    ret, frame = cap.read()
    print(frame.shape)
    rescaled_frame = rescale_frame(scale_factor, frame)
    (h, w) = rescaled_frame.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter('vid_new.mp4',
                             fourcc, 15.0,
                             (w, h), True)
else:
    print("Camera is not opened")

while cap.isOpened():
    ret, frame = cap.read()

    rescaled_frame = rescale_frame(scale_factor, frame)

    # write the output frame to file
    writer.write(rescaled_frame)

    cv2.imshow("Output", rescaled_frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break


cv2.destroyAllWindows()
cap.release()
writer.release()