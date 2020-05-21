import numpy as np

from .custom import CustomDataset


class PEDataset(CustomDataset):

    CLASSES = ('sate',)

    def __init__(self, **kwargs):
	super(PEDataset, self).__init__(**kwargs)
