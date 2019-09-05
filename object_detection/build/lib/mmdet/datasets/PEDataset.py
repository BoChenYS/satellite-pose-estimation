import numpy as np
import mmcv

from .custom import CustomDataset


class PEDataset(CustomDataset):

    CLASSES = ('sate',)

    def __init__(self, **kwargs):
        super(PEDataset, self).__init__(**kwargs)

    def load_annotations(self, ann_file):
        ann = mmcv.load(ann_file)
        new_ann = []
        for rec in ann:
            new_ann.append({
                'filename': rec['filename'],
                'width': rec['width'],
                'height': rec['height'],
                'ann': {
                    'bboxes': np.array(rec['ann']['bboxes'], dtype=np.float32),
                    'labels': np.array(rec['ann']['labels'], dtype=np.long),
                    }
            })
        return new_ann
