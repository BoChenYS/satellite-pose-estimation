import numpy as np
import mmcv
import json

from .custom import CustomDataset


class PEDataset(CustomDataset):

    CLASSES = ('sate',)

    def __init__(self, **kwargs):
        super(PEDataset, self).__init__(**kwargs)

    # def load_annotations(self, ann_file):
    #     ann = mmcv.load(ann_file)
    #     new_ann = []
    #     for rec in ann:
    #         new_ann.append({
    #             'filename': rec['filename'],
    #             'width': rec['width'],
    #             'height': rec['height'],
    #             'ann': {
    #                 'bboxes': np.array(rec['ann']['bboxes'], dtype=np.float32),
    #                 'labels': np.array(rec['ann']['labels'], dtype=np.long),
    #                 }
    #         })
    #     return new_ann

    def load_annotations(self, ann_file):

        ann = mmcv.load(ann_file)
        new_ann = []
        for rec in ann:
            new_ann.append({
                'filename': rec['image'],
                'width': 1920,
                'height': 1200,
                'ann': {
                    'bboxes': np.array(rec['box'], dtype=np.float32),
                    'labels': np.array([1], dtype=np.long),
                }
            })

        # ann_file2 = ann_file + '_coco.json'
        # with open(ann_file2, 'w') as f:
        #     new_ann2 = []
        #     for rec in new_ann:
        #         new_rec = rec.copy()
        #         new_rec['ann']['bboxes'] = new_rec['ann']['bboxes'].tolist()
        #         new_rec['ann']['labels'] = new_rec['ann']['labels'].tolist()
        #         new_ann2.append(new_rec)
        #     # mmcv.dump(new_ann2, f, file_format='json')
        #     json.dump(new_ann2, f)
        # from pycocotools.coco import COCO
        # self.coco = COCO(ann_file2)

        return new_ann




