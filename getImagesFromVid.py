import cv2

def getFrameCount():
    cap = cv2.VideoCapture("vid.mp4")
    frameCount = 0
    while cap.isOpened():
        ret, _ = cap.read()
        if ret:
            frameCount += 1
        else:
            cap.release()
            break

    return frameCount

cap = cv2.VideoCapture("vid.mp4")
currentFrameNum = 0
totalFrameCount = getFrameCount()
numImages = 100
frameInterval = totalFrameCount // numImages

while cap.isOpened():
    ret, frame = cap.read()
    if ret:
        # cv2.imshow("vid", frame)
        if currentFrameNum % frameInterval == 0:
            cv2.imwrite('./images/image_' + str(currentFrameNum//frameInterval) + ".jpg", frame)
        currentFrameNum += 1
        if cv2.waitKey(1) == ord('q'):
            break
    else:
        cap.release()
        break

print("Images created!")

cv2.destroyAllWindows()

