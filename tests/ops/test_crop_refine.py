import numpy as np

from fakes import SmallInputSegmenter
from vision_toolkit.ops.crop_refine import bounding_box, segment_on_crop
from vision_toolkit.tasks.segmentation import Segmenter


def small_subject():
    image = np.zeros((400, 400, 3), np.uint8)
    image[200:206, 300:306] = 255  # a 6x6 "person"
    region = np.zeros((400, 400), np.uint8)
    region[200:206, 300:306] = 255
    return image, region


def test_bounding_box_margin_and_clipping():
    mask = np.zeros((100, 100), np.uint8)
    mask[10:20, 90:100] = 255
    assert bounding_box(mask) == (90, 10, 100, 20)
    assert bounding_box(mask, margin=0.5) == (86, 6, 100, 24)
    assert bounding_box(np.zeros((5, 5), np.uint8)) is None


def test_small_input_model_finds_subject_only_on_crop():
    image, region = small_subject()
    segmenter = SmallInputSegmenter()
    assert isinstance(segmenter, Segmenter)

    assert segmenter(image).mask_of("person").sum() == 0
    refined = segment_on_crop(segmenter, image, region, margin=0.25)
    assert refined.shape == region.shape
    assert (refined[region > 0] == 255).mean() > 0.8
    assert (refined[region == 0] == 255).sum() < 20


def test_empty_region_gives_empty_mask():
    image, _ = small_subject()
    out = segment_on_crop(SmallInputSegmenter(), image, np.zeros((400, 400), np.uint8))
    assert out.shape == (400, 400) and not out.any()
