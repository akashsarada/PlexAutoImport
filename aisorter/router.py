import os
import shutil

KNOWN_CATEGORIES: frozenset[str] = frozenset({'cars', 'people', 'scenery'})


class PhotoRouter:
    def __init__(self, output_root: str) -> None:
        self.output_root = output_root

    def _ensure_dir(self, path: str) -> str:
        os.makedirs(path, exist_ok=True)
        return path

    def route(self, image_path: str, labels: list[str]) -> list[str]:
        filename = os.path.basename(image_path)
        label_set = set(labels)
        identity_names = [l for l in labels if l not in KNOWN_CATEGORIES]

        destinations: list[str] = []

        for name in identity_names:
            destinations.append(
                os.path.join(self._ensure_dir(os.path.join(self.output_root, 'sorted', 'people', name)), filename)
            )

        if not identity_names and 'people' in label_set:
            destinations.append(
                os.path.join(self._ensure_dir(os.path.join(self.output_root, 'sorted', 'people', 'unknown')), filename)
            )

        if 'cars' in label_set:
            destinations.append(
                os.path.join(self._ensure_dir(os.path.join(self.output_root, 'sorted', 'cars')), filename)
            )

        if 'scenery' in label_set:
            destinations.append(
                os.path.join(self._ensure_dir(os.path.join(self.output_root, 'sorted', 'scenery')), filename)
            )

        if not destinations:
            destinations.append(
                os.path.join(self._ensure_dir(os.path.join(self.output_root, 'sorted', 'misc')), filename)
            )

        for dest in destinations[:-1]:
            shutil.copy2(image_path, dest)

        shutil.move(image_path, destinations[-1])

        return destinations
