"""
Joint Data Augmentation for Polyp Segmentation with Small-Lesion Enhancement.

Features:
1. Synchronized spatial transforms: Random horizontal/vertical flip, discrete rotation.
2. Scale Jittering (Zoom-in / Zoom-out) to simulate lesions at diverse camera distances.
3. Diminutive Lesion Copy-Paste (Online Small-Lesion Injector):
   - Extracts existing lesions, scales them down to diminutive sizes (0.25x - 0.5x),
     and seamlessly pastes them onto unmasked background areas.
   - Inoculates the network against missing tiny, flat polyps (characteristic of ETIS-LaribPolypDB).
"""

import random
import numpy as np
from PIL import Image
import torchvision.transforms.functional as TF


class PolypAugmentation:
    """Joint random transformations for image and ground truth mask."""

    def __init__(
        self,
        hflip_prob: float = 0.5,
        vflip_prob: float = 0.5,
        rotate_prob: float = 0.5,
        scale_prob: float = 0.3,
        copypaste_prob: float = 0.35,
        angles: tuple = (90, 180, 270),
    ):
        self.hflip_prob = hflip_prob
        self.vflip_prob = vflip_prob
        self.rotate_prob = rotate_prob
        self.scale_prob = scale_prob
        self.copypaste_prob = copypaste_prob
        self.angles = angles

    def _small_object_copypaste(self, image: Image.Image, mask: Image.Image) -> tuple[Image.Image, Image.Image]:
        """
        Extracts lesion, scales it to diminutive size, and pastes into background.
        Directly targets the weakness on ETIS diminutive polyps.
        """
        m_np = np.array(mask)
        pos = np.argwhere(m_np > 128)
        # Only copy-paste if there is a valid lesion of reasonable size
        if pos.shape[0] < 200:
            return image, mask

        ymin, xmin = pos.min(axis=0)
        ymax, xmax = pos.max(axis=0)
        w_box = xmax - xmin
        h_box = ymax - ymin

        if w_box < 10 or h_box < 10:
            return image, mask

        # Crop lesion patch
        crop_img = image.crop((xmin, ymin, xmax, ymax))
        crop_m = mask.crop((xmin, ymin, xmax, ymax))

        # Scale down to diminutive lesion (0.25x - 0.55x)
        scale = random.uniform(0.25, 0.55)
        new_w = max(12, int(w_box * scale))
        new_h = max(12, int(h_box * scale))
        c_img_scaled = crop_img.resize((new_w, new_h), Image.BILINEAR)
        c_m_scaled = crop_m.resize((new_w, new_h), Image.NEAREST)

        # Optional random rotation for the pasted lesion
        rot = random.choice([0, 90, 180, 270])
        if rot > 0:
            c_img_scaled = c_img_scaled.rotate(rot, expand=True)
            c_m_scaled = c_m_scaled.rotate(rot, expand=True)
            new_w, new_h = c_img_scaled.size

        # Find random background coordinate (where mask is 0)
        w_full, h_full = image.size
        max_x = max(0, w_full - new_w)
        max_y = max(0, h_full - new_h)
        if max_x == 0 or max_y == 0:
            return image, mask

        for _ in range(5):  # Try up to 5 times to find background spot
            px = random.randint(0, max_x)
            py = random.randint(0, max_y)
            # Check overlap with existing lesion
            sub_m = m_np[py:py + new_h, px:px + new_w]
            if (sub_m > 128).mean() < 0.05:  # Overwhelmingly background
                img_copy = image.copy()
                mask_copy = mask.copy()
                img_copy.paste(c_img_scaled, (px, py), mask=c_m_scaled)
                mask_copy.paste(c_m_scaled, (px, py), mask=c_m_scaled)
                return img_copy, mask_copy

        return image, mask

    def __call__(self, image: Image.Image, mask: Image.Image):
        # 1. Random Horizontal Flip
        if random.random() < self.hflip_prob:
            image = TF.hflip(image)
            mask = TF.hflip(mask)

        # 2. Random Vertical Flip
        if random.random() < self.vflip_prob:
            image = TF.vflip(image)
            mask = TF.vflip(mask)

        # 3. Random Discrete Rotation
        if random.random() < self.rotate_prob:
            angle = random.choice(self.angles)
            image = TF.rotate(image, angle)
            mask = TF.rotate(mask, angle)

        # 4. Random Scale Jittering (Zoom in / Zoom out)
        if random.random() < self.scale_prob:
            scale_factor = random.uniform(0.8, 1.2)
            w, h = image.size
            new_w, new_h = int(w * scale_factor), int(h * scale_factor)
            image_rescaled = TF.resize(image, [new_h, new_w], interpolation=TF.InterpolationMode.BILINEAR)
            mask_rescaled = TF.resize(mask, [new_h, new_w], interpolation=TF.InterpolationMode.NEAREST)

            if scale_factor > 1.0:
                # Center crop back to (w, h)
                image = TF.center_crop(image_rescaled, [h, w])
                mask = TF.center_crop(mask_rescaled, [h, w])
            else:
                # Pad back to (w, h)
                pad_left = (w - new_w) // 2
                pad_top = (h - new_h) // 2
                pad_right = w - new_w - pad_left
                pad_bottom = h - new_h - pad_top
                image = TF.pad(image_rescaled, [pad_left, pad_top, pad_right, pad_bottom], fill=0)
                mask = TF.pad(mask_rescaled, [pad_left, pad_top, pad_right, pad_bottom], fill=0)

        # 5. Online Diminutive Lesion Copy-Paste (Targets small ETIS polyps)
        if random.random() < self.copypaste_prob:
            image, mask = self._small_object_copypaste(image, mask)

        return image, mask
