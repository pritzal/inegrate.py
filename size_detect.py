import cv2

# load the image
img = cv2.imread('object_image.jpg')

# convert the image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# apply Gaussian blur to reduce noise
blur = cv2.GaussianBlur(gray, (5, 5), 0)

# perform Canny edge detection
edges = cv2.Canny(blur, 50, 150)

# perform morphological operations to close gaps and remove noise
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

# find contours in the image
contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# iterate over each contour
for contour in contours:
    # calculate the area of the contour
    area = cv2.contourArea(contour)
    
    # calculate the perimeter of the contour
    perimeter = cv2.arcLength(contour, True)
    
    # calculate the circularity of the contour
    circularity = 4 * 3.14 * area / (perimeter ** 2)
    
    # print the size information of the contour
    print('Area:', area)
    print('Perimeter:', perimeter)
    print('Circularity:', circularity)
    
    # draw the contour on the image
    cv2.drawContours(img, [contour], 0, (0, 255, 0), 2)

# show the image with the contour drawn
cv2.imshow('Object Detection', img)

# wait for a key press and then exit
cv2.waitKey(0)
cv2.destroyAllWindows()
