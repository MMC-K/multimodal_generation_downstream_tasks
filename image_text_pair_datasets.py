from builtins import isinstance
import os
import glob
import json
import logging
import zipfile
import functools

import datasets

logger = logging.getLogger(__name__)

_VERSION = datasets.Version("1.0.0", "")

_URL = ""

_CITATION = """\
There is no citation information
"""

_DESCRIPTION = """\
image text pair datasets
"""


IMAGE_TEXT_PAIR_DEFAULT_FEATURES=datasets.Features(
                {
                    "image": datasets.Image(),
                    "description": datasets.Value("string"),
                    "image_url": datasets.Value("string"),
                }
            )


def generator(fname, root_dir):
    for item in json.load(open(fname, "r")):
        #print("[BG]***", item, root_dir)
        if len(item["file_name"]) > 10:
            image_path = os.path.join(root_dir, item["file_name"][:3], item["file_name"][3:6], item["file_name"])
        else:
            image_path = os.path.join(root_dir, item["file_name"][:2], item["file_name"])
        #image_path = os.path.join(root_dir, item["path"])
        description = item["description"]
        image_url = item["image_url"] if "image_url" in item.keys() else None
        yield {
            "image": {
                "path": image_path,
                "bytes": open(image_path, "rb").read(),
            },
            "description": description,
            "image_url": image_url,
        }


class CustomBuilderConfig(datasets.BuilderConfig):
    def __init__(self, **kwargs):
        self.train_csv_path  = None #sv_path
        self.validation_csv_path  = None #sv_path
        super().__init__(name="base", version=_VERSION, description="Image Text Pair Dataset", **kwargs)

class ImageTextPairDataset(datasets.GeneratorBasedBuilder):
    """Image Text Pair Dataset"""

    BUILDER_CONFIGS = [
            CustomBuilderConfig()
    ]

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=IMAGE_TEXT_PAIR_DEFAULT_FEATURES,
            supervised_keys=None,  # Probably needs to be fixed.
            homepage=_URL,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager: datasets.DownloadManager):

        path_kv = {
            datasets.Split.TRAIN: [self.config.train_csv_path, dl_manager.manual_dir],
            datasets.Split.VALIDATION: [self.config.validation_csv_path, dl_manager.manual_dir],
        }

        return [
                datasets.SplitGenerator(name=k, gen_kwargs={'fpath': v, 'root_dir': vv}) for k, (v, vv) in path_kv.items() if v is not None
        ]

    def _generate_examples(self, fpath, root_dir):
        """Yields examples."""
        for idx, item in enumerate(generator(fpath, root_dir)):
            yield idx, item

