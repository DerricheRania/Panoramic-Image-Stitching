#reference: https://pylessons.com/OpenCV-image-stiching-continue

import cv2
import numpy as np
import matplotlib.pyplot as plt

def trim(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)

    top, bottom, left, right = 0, frame.shape[0], 0, frame.shape[1]

    while top < bottom and np.all(thresh[top, left:right] == 0):        
        top += 1
    while bottom > top and np.all(thresh[bottom-1, left:right] == 0):
        bottom -= 1
    while left < right and np.all(thresh[top:bottom, left] == 0):
        left += 1
    while right > left and np.all(thresh[top:bottom, right-1] == 0):
        right -= 1

    return frame[top:bottom, left:right]

image_files = ['img1.jpg', 'img2.jpg','img3.jpg'] 
images = [cv2.imread(f) for f in image_files]

result = images[0]

for i in range(1, len(images)):   

    img_l = result
    img_r = images[i]

    imgr = cv2.cvtColor(img_r, cv2.COLOR_BGR2GRAY)
    imgl = cv2.cvtColor(img_l, cv2.COLOR_BGR2GRAY)

    sift = cv2.SIFT_create()
    kpr, desr = sift.detectAndCompute(imgr, None)
    kpl, desl = sift.detectAndCompute(imgl, None)
    plt.figure(figsize=(10,6))
    plt.imshow(cv2.cvtColor(cv2.drawKeypoints(img_l, kpl, None), cv2.COLOR_BGR2RGB))
    plt.title('original_image_left_keypoints')
    plt.axis('off')
    plt.show()

    plt.figure(figsize=(10,6))
    plt.imshow(cv2.cvtColor(cv2.drawKeypoints(img_r, kpr, None), cv2.COLOR_BGR2RGB))
    plt.title('original_image_right_keypoints')
    plt.axis('off')
    plt.show()

    match = cv2.BFMatcher()
    matches = match.knnMatch(desr, desl, k=2)

    good = []
    for m, n in matches:
        if m.distance < 0.75*n.distance:
            good.append(m)

    draw_params = dict(matchColor=(0,255,0),
                           singlePointColor=None,
                           flags=2)

    img3 = cv2.drawMatches(img_r, kpr, img_l, kpl, good, None, **draw_params)
    plt.figure(figsize=(14,6))
    plt.imshow(cv2.cvtColor(img3, cv2.COLOR_BGR2RGB))
    plt.title('Draw Matches Left Right')
    plt.axis('off')
    plt.show()

    MIN_MATCH_COUNT = 10
    if len(good) > MIN_MATCH_COUNT:
        src_pts = np.float32([ kpr[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
        dst_pts = np.float32([ kpl[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

        h, w = imgr.shape
        pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
        dst = cv2.perspectiveTransform(pts, M)
        imgr = cv2.polylines(imgr, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)
        plt.figure(figsize=(10,6))
        plt.imshow(imgr, cmap='gray')
        plt.title('right_image_overlapping')
        plt.axis('off')
        plt.show()
    else:
        print("Not enought matches are found - %d/%d", (len(good)/MIN_MATCH_COUNT))
        continue

    print(img_l.shape[1] + img_r.shape[1], img_l.shape[0])
    h_new = img_l.shape[0] + img_r.shape[0]   
    w_new = img_l.shape[1] + img_r.shape[1]
    dst = cv2.warpPerspective(img_r, M, (w_new, h_new))
    plt.figure(figsize=(14,6))
    plt.imshow(cv2.cvtColor(dst, cv2.COLOR_BGR2RGB))
    plt.title('dst warpPerspective')
    plt.axis('off')
    plt.show() 


    mask_black = np.all(dst[0:img_l.shape[0], 0:img_l.shape[1]] == 0, axis=2)
    roi = dst[0:img_l.shape[0], 0:img_l.shape[1]]
    roi[mask_black] = img_l[mask_black]
    dst[0:img_l.shape[0], 0:img_l.shape[1]] = roi
    plt.figure(figsize=(14,6))
    plt.imshow(cv2.cvtColor(dst, cv2.COLOR_BGR2RGB))
    plt.title('add left image to the warped right')
    plt.axis('off')
    plt.show()

    result = dst

    result = trim(dst)
    cv2.imwrite(f"result_step_{i}.jpg", result)
    print(f"Saved result_step_{i}.jpg")

arr = np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])
print(arr[:, -1])
print(arr[:, :-2])
print(arr[:,0])
print(arr[:,1:])
print(arr[-1])
print(arr[0:-1])
print(arr[0])
print(arr[1:])
final_panorama = trim(result)
cv2.imwrite("final_result.jpg", final_panorama)
print("Full panorama saved!")
plt.figure(figsize=(18,6))
plt.imshow(cv2.cvtColor(final_panorama, cv2.COLOR_BGR2RGB))
plt.title('FINAL PANORAMIC IMAGE')
plt.axis('off')
plt.show()