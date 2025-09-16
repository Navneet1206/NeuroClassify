import os
import argparse
from pathlib import Path
import numpy as np
import cv2

CLASSES = ["glioma", "meningioma", "pituitary", "normal"]


def make_pattern(cls: str, h: int = 256, w: int = 256) -> np.ndarray:
    img = np.zeros((h, w, 3), dtype=np.uint8)
    rng = np.random.default_rng()

    # Background noise
    noise = rng.integers(0, 30, size=(h, w, 3), dtype=np.uint8)
    img = cv2.add(img, noise)

    # Draw class-specific patterns
    if cls == "glioma":
        # irregular blob with random contour
        pts = rng.integers(0, min(h, w), size=(rng.integers(5, 12), 2))
        hull = cv2.convexHull(pts.astype(np.int32))
        cv2.fillConvexPoly(img, hull, (int(rng.integers(100, 255)), 0, 0))
    elif cls == "meningioma":
        # concentric circles
        center = (w // 2 + rng.integers(-20, 20), h // 2 + rng.integers(-20, 20))
        for r in range(20, min(h, w) // 2, 20):
            cv2.circle(img, center, r, (0, int(rng.integers(100, 255)), 0), 2)
    elif cls == "pituitary":
        # small bright ellipse near center
        center = (w // 2 + rng.integers(-10, 10), h // 2 + rng.integers(-10, 10))
        axes = (rng.integers(10, 25), rng.integers(6, 18))
        cv2.ellipse(
            img,
            (int(center[0]), int(center[1])),
            (int(axes[0]), int(axes[1])),
            int(rng.integers(0, 180)),
            0,
            360,
            (0, 0, int(rng.integers(180, 255))),
            -1,
        )
    elif cls == "normal":
        # soft gradient background
        gx = np.tile(np.linspace(0, 60, w, dtype=np.uint8), (h, 1))
        gy = np.tile(np.linspace(0, 60, h, dtype=np.uint8), (w, 1)).T
        grad = cv2.merge([gx, gy, ((gx + gy) // 2).astype(np.uint8)])
        img = cv2.add(img, grad)

    # Add mild blur and brightness variation
    img = cv2.GaussianBlur(img, (5, 5), 0)
    img = cv2.convertScaleAbs(img, alpha=1.0, beta=int(rng.integers(-10, 10)))

    # Optional text label (weak imprint) for visibility
    cv2.putText(img, cls[:3].upper(), (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1, cv2.LINE_AA)
    return img


def main():
    ap = argparse.ArgumentParser(description="Generate dummy synthetic images for quick pipeline tests.")
    ap.add_argument("--out", default="data/raw", help="Output root directory where class folders will be created.")
    ap.add_argument("--n", type=int, default=100, help="Number of images per class.")
    ap.add_argument("--size", type=int, default=256, help="Image size (square).")
    args = ap.parse_args()

    root = Path(args.out)
    for cls in CLASSES:
        (root / cls).mkdir(parents=True, exist_ok=True)
    for cls in CLASSES:
        print(f"Generating {args.n} images for {cls}...")
        for i in range(args.n):
            img = make_pattern(cls, args.size, args.size)
            out_path = root / cls / f"{cls}_{i:05d}.png"
            cv2.imwrite(str(out_path), img)
    print(f"Done. Images saved under {root}.")


if __name__ == "__main__":
    main()
