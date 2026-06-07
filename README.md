# 🖼️ Panoramic Image Stitching 

## Overview

This project builds a **panoramic image** by stitching together multiple overlapping photographs taken with a small horizontal shift (translation to the right) between each shot. Using classical computer vision techniques — SIFT feature detection, feature matching, and homography estimation — the images are aligned and blended into a single wide panorama.

---

## How the Images Were Taken

The input images were captured by keeping the camera fixed in position and rotating/shifting it slightly to the **right** between each shot, so that consecutive images share a visible overlapping region. This overlap is essential: it gives the algorithm enough shared visual content to figure out how the images relate to each other geometrically.

- **Number of images:** Any number ≥ 2 (3 were used in this demo: `img1.jpg`, `img2.jpg`, `img3.jpg`)
- **Direction of shift:** Left → Right
- **Overlap:** Each consecutive pair of images shares a portion of the scene

---

## Processing Pipeline

The stitching is done incrementally: the first image is used as the base, and each subsequent image is aligned and merged into the growing result.

### Step 1 — Grayscale Conversion

Each image pair (the current result and the next image) is converted to grayscale. Color information is not needed for feature detection, and working in grayscale reduces computation.

### Step 2 — SIFT Feature Detection

**SIFT (Scale-Invariant Feature Transform)** is applied to both images to detect *keypoints* — distinctive local regions in the image (corners, edges, blobs) that are robust to scale changes, rotation, and minor perspective distortions.

For each keypoint, a 128-dimensional **descriptor** vector is computed, encoding the local appearance around that point. These descriptors are what allow matching between images.

> Keypoint visualizations are displayed for both the left and right images at this stage.

### Step 3 — Feature Matching (BFMatcher + Lowe's Ratio Test)

A **Brute-Force Matcher (BFMatcher)** compares descriptors from the right image against descriptors from the left image using k-Nearest Neighbors (k=2).

To filter out poor matches, **Lowe's ratio test** is applied:

```
if m.distance < 0.75 × n.distance → keep match
```

This keeps only matches where the best candidate is significantly closer than the second-best, eliminating ambiguous correspondences.

> The accepted matches are visualized with green lines drawn between the two images.

### Step 4 — Homography Estimation (RANSAC)

A **homography** is a 3×3 transformation matrix that describes how to warp one image so it aligns with the other. It is computed from the good matches using **RANSAC (Random Sample Consensus)**, an algorithm that is robust to outliers (incorrectly matched points).

At least **10 good matches** are required for this step to proceed.

The right image's boundary is then projected onto the left image's coordinate space to visualize the overlap region.

### Step 5 — Perspective Warping

The right image is warped using `cv2.warpPerspective` with the computed homography matrix, projecting it into the coordinate frame of the left (base) image.

A canvas large enough to hold both images is allocated:

```
new width  = width_left  + width_right
new height = height_left + height_right
```

> The warped right image is displayed at this stage.

### Step 6 — Image Blending

The left image is placed onto the warped canvas. To avoid overwriting valid pixels from the right image with black background pixels, a **black-pixel mask** is used:

- Pixels in the canvas region corresponding to the left image that are still black (from the warp) are replaced with the actual left image pixels.
- This produces a simple but effective blend where the two images share overlapping content seamlessly.

> The combined result is displayed after blending.

### Step 7 — Trimming

The `trim()` function removes any black borders introduced by the warping process by scanning the image from all four sides (top, bottom, left, right) and cropping to the tightest bounding box that contains actual image content.

### Step 8 — Saving Intermediate Results

After each stitching step, the current result is saved:

```
result_step_1.jpg  ← after stitching img1 + img2
result_step_2.jpg  ← after stitching result + img3
result_step_N.jpg  ← one file per additional image
```

---

## Final Output

After all images are processed and the final trim is applied, the panorama is saved as:

```
final_result.jpg
```

This is the complete wide-angle panoramic image assembled from all three input shots.

---

## Dependencies

| Library | Purpose |
|---|---|
| `opencv-python` | Image I/O, SIFT, homography, warping |
| `numpy` | Array and matrix operations |
| `matplotlib` | Visualization of intermediate steps |

Install with:

```bash
pip install opencv-python numpy matplotlib
```

---

## File Structure

```
project/
├── img1.jpg               # Input image 1 (leftmost)
├── img2.jpg               # Input image 2
├── img3.jpg               # Input image N (rightmost) — add as many as needed
├── Miniproject.py         # Main stitching script
├── result_step_1.jpg      # Intermediate result after stitching image 2
├── result_step_2.jpg      # Intermediate result after stitching image 3
├── result_step_N.jpg      # One file saved per stitching step
└── final_result.jpg       # Final panoramic image ✅
```

---

## Algorithm Summary

```
img1 ──┐
       ├─► SIFT → Match → Homography → Warp → Blend → Trim ──┐
img2 ──┘                                                       │
                                                    result_step_1
                                                               │
       ┌───────────────────────────────────────────────────── ┘
       ├─► SIFT → Match → Homography → Warp → Blend → Trim ──┐
img3 ──┘                                                       │
                                                    result_step_2
                                                               │
                                     ...  (repeats for each additional image)
                                                               │
                                                    final_result.jpg ✅
```

---

## References

- [OpenCV Image Stitching — PyLessons](https://pylessons.com/OpenCV-image-stiching-continue)
- Lowe, D.G. (2004). *Distinctive Image Features from Scale-Invariant Keypoints*. IJCV.
- Fischler & Bolles (1981). *Random Sample Consensus (RANSAC)*. Communications of the ACM.
