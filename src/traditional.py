import cv2
import numpy as np


def cartoonize(image: np.ndarray,
               bilateral_d: int = 9,
               sigma_color: float = 75,
               sigma_space: float = 75,
               canny_low: int = 50,
               canny_high: int = 150,
               k_colors: int = 12) -> np.ndarray:
    bilateral = cv2.bilateralFilter(image, bilateral_d, sigma_color, sigma_space)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, canny_low, canny_high)
    edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    edges_inv = 255 - edges

    data = bilateral.reshape((-1, 3)).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    _, labels, centers = cv2.kmeans(data, k_colors, None, criteria, 10,
                                     cv2.KMEANS_PP_CENTERS)
    centers = centers.astype(np.uint8)
    quantized = centers[labels.flatten()].reshape(bilateral.shape)

    result = cv2.bitwise_and(quantized, edges_inv)
    return result


def pencil_sketch(image: np.ndarray,
                  blur_kernel: int = 21) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_inv = 255 - gray
    blurred = cv2.GaussianBlur(gray_inv, (blur_kernel, blur_kernel), sigmaX=0, sigmaY=0)
    sketch = cv2.divide(gray.astype(np.float32), 255 - blurred.astype(np.float32), scale=256.0)
    sketch = np.clip(sketch, 0, 255).astype(np.uint8)
    return cv2.cvtColor(sketch, cv2.COLOR_GRAY2BGR)


def watercolor(image: np.ndarray,
               mean_shift_sp: int = 15,
               mean_shift_sr: int = 40,
               gamma: float = 1.2,
               saturation_boost: float = 1.1,
               noise_sigma: float = 3.0) -> np.ndarray:
    ms = cv2.pyrMeanShiftFiltering(image, mean_shift_sp, mean_shift_sr)

    hsv = cv2.cvtColor(ms, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 2] = 255.0 * ((hsv[:, :, 2] / 255.0) ** gamma)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * saturation_boost, 0, 255)
    toned = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    noise = np.random.normal(0, noise_sigma, toned.shape).astype(np.float32)
    result = toned.astype(np.float32) + noise
    result = np.clip(result, 0, 255).astype(np.uint8)

    return result


def apply_traditional(image: np.ndarray, style: str) -> np.ndarray:
    if style == "cartoon":
        return cartoonize(image)
    elif style == "sketch":
        return pencil_sketch(image)
    elif style == "watercolor":
        return watercolor(image)
    else:
        raise ValueError(f"Unknown style: {style}")


if __name__ == "__main__":
    import sys
    cv2.setRNGSeed(42)
    if len(sys.argv) < 3:
        print("Usage: python traditional.py <input_image> <style: cartoon|sketch|watercolor> [output]")
        sys.exit(1)

    img = cv2.imread(sys.argv[1])
    if img is None:
        print(f"Error loading {sys.argv[1]}")
        sys.exit(1)

    result = apply_traditional(img, sys.argv[2])
    out_path = sys.argv[3] if len(sys.argv) > 3 else f"{sys.argv[2]}_output.jpg"
    cv2.imwrite(out_path, result)
    print(f"Saved to {out_path}")
