import os
from pathlib import Path
from PIL import Image
import numpy as np
import torch
import torchvision.transforms.functional as TF

class MODSDataset(torch.utils.data.Dataset):
    """MODS dataset wrapper

    Args:
        seq_mapping (str): Path to the txt file, containing mods sequence -> subdir mappings
        transform (optional): Tranform to apply to image and masks
    """

    def __init__(self, seq_mapping, transform=None, normalize_t=None):
        seq_mapping = Path(seq_mapping)
        base_dir = seq_mapping.parent

        data = []
        with seq_mapping.open() as file:
            seq_pairs = (tuple(l.split()) for l in file)

            for modd_seq, seq in seq_pairs:
                seq_dir = base_dir / seq
                modd_mapping_fn = seq_dir / 'mapping.txt'
                imu_mapping_fn = seq_dir / 'imu_mapping.txt'

                seq_data = {}
                with (seq_dir / 'imu_mapping.txt').open() as file:
                    imu_pairs = (tuple(l.split()) for l in file)
                    for img_fn, imu_fn in imu_pairs:
                        img_path = seq_dir / img_fn
                        imu_path = seq_dir / imu_fn
                        seq_data[img_path.name] = {
                            'image_path': str(img_path),
                            'imu_path': str(imu_path),
                            'name': img_path.name,
                            'modd_seq': modd_seq,
                            'seq': seq
                        }

                with (seq_dir / 'mapping.txt').open() as file:
                    pairs = (tuple(l.split()) for l in file)
                    for img_name, modd_name in pairs:
                        seq_data[img_name]['modd_name'] = modd_name

                data.extend(seq_data.values())

        self.data = data
        self.transform = transform
        self.normalize_t = normalize_t

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        entry = self.data[idx]

        img = np.array(Image.open(entry['image_path']))
        imu = np.array(Image.open(entry['imu_path']))

        data = {
            'image': img,
            'imu_mask': imu
        }

        # Transform images and masks if transform is provided
        if self.transform is not None:
            transformed = self.transform(data)
            img = transformed['image']
            imu = transformed['imu_mask']

        if self.normalize_t is not None and self.normalize_t != "skip":
            img = self.normalize_t(img)
        elif self.normalize_t == "skip":
            pass
        else:
            # Default: divide by 255
            img = TF.to_tensor(img)
        imu = torch.from_numpy(imu.astype(np.bool))

        metadata_fields = ['seq', 'name', 'modd_seq', 'modd_name']
        metadata = {field: entry[field] for field in metadata_fields}
        metadata['image_path'] = os.path.join(metadata['seq'], metadata['name'])

        features ={
            'image': img,
            'imu_mask': imu,
        }

        return features, metadata
