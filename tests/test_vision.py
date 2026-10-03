import cv2
import numpy as np

from catalarm.config import MotionConfig, Thresholds
from catalarm.vision.detector import Detection, FullFrameDetector
from catalarm.vision.embedder import HistogramEmbedder, l2norm, preprocess
from catalarm.vision.matcher import MATCH, NO_CAT, UNCONFIRMED, match
from catalarm.vision.motion import MotionDetector
from catalarm.vision.nightvision import is_night_vision
from catalarm.vision.tracker import NO_MOTION, Tracker

from .conftest import TEMPLATES, TEST_IMG


def vec(*v):
    return l2norm(np.array(v, np.float32))


def test_match_ranks_and_thresholds_on_same_score():
    refs = np.stack([vec(1, 0, 0), vec(0.9, 0.1, 0), vec(0, 1, 0)])
    labels = ["maugli", "maugli", "gita"]
    r = match(vec(1, 0.02, 0), refs, labels, Thresholds())
    assert r.state == MATCH and r.cat_id == "maugli"
    assert list(r.distances) == ["maugli", "gita"]
    assert r.distance == r.distances["maugli"]
    assert r.margin > 0.5


def test_match_requires_margin():
    refs = np.stack([vec(1, 0.05, 0), vec(1, -0.05, 0)])
    r = match(vec(1, 0, 0), refs, ["a", "b"], Thresholds(margin=0.05))
    assert r.state == UNCONFIRMED


def test_match_far_is_no_cat_and_empty_gallery():
    refs = np.stack([vec(1, 0, 0)])
    assert match(vec(0, 0, 1), refs, ["a"], Thresholds()).state == NO_CAT
    assert match(vec(0, 0, 1), np.zeros((0, 3)), [], Thresholds()).state == NO_CAT


def result(state, cat="maugli", d=0.1):
    from catalarm.vision.matcher import MatchResult

    return MatchResult(state, cat, d, 0.3, {cat: d})


def test_tracker_votes_and_holds():
    t = Tracker(Thresholds(vote_k=3, vote_n=5, hold_s=10))
    assert t.update(0, False).state == NO_MOTION
    # strong matches still collecting votes are not reported as "unconfirmed"
    assert t.update(1, True, result(MATCH)).state == "no_cat"
    assert t.update(2, True, result(MATCH)).state == "no_cat"
    s = t.update(3, True, result(MATCH))
    assert s.state == "maugli" and s.changed and s.is_cat
    # a single stray frame of another cat does not flip the state
    assert t.update(4, True, result(MATCH, "gita")).state == "maugli"
    # maugli still had 3 of 5 votes at t=4; nothing seen since: hold until t=14
    assert t.update(12, True).state == "maugli"
    assert t.update(14.5, True).state == "no_cat"
    assert t.update(30, False).state == NO_MOTION


def test_tracker_weak_matches_are_unconfirmed():
    t = Tracker(Thresholds(vote_k=2, vote_n=3, hold_s=10))
    assert t.update(0, True, result(UNCONFIRMED, d=0.4)).state == "unconfirmed"
    assert t.update(1, True, result(MATCH)).state == "unconfirmed"
    assert t.update(2, True, result(MATCH)).state == "maugli"


def test_motion_detector_ignores_masked_area():
    frame = np.zeros((360, 640, 3), np.uint8)
    md = MotionDetector(MotionConfig(mask=[[(0, 0), (0.5, 0), (0.5, 1), (0, 1)]]))
    for _ in range(20):
        md.apply(frame)
    moved = frame.copy()
    cv2.rectangle(moved, (20, 100), (200, 300), (255, 255, 255), -1)  # left half: masked
    assert not md.apply(moved).detected
    moved2 = frame.copy()
    cv2.rectangle(moved2, (400, 100), (600, 300), (255, 255, 255), -1)
    assert md.apply(moved2).detected


def test_night_vision_detection():
    color = cv2.imread(str(TEST_IMG / "maugli1.jpg"))
    assert not is_night_vision(color)
    night = cv2.imread(str(next((TEMPLATES / "NightTemplates").glob("*.png"))))
    assert is_night_vision(night)
    assert is_night_vision(cv2.cvtColor(cv2.cvtColor(color, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR))


def test_detection_crop_with_padding_is_clamped():
    frame = np.zeros((100, 200, 3), np.uint8)
    crop = Detection((0, 0, 100, 50), 0.9).crop(frame, padding=0.5)
    assert crop.shape[:2] == (62, 125)
    assert FullFrameDetector().detect(frame)[0].box == (0, 0, 200, 100)


def test_embedders_are_normalized():
    img = cv2.imread(str(TEST_IMG / "gita1.jpg"))
    e = HistogramEmbedder().embed([img, img])
    assert e.shape == (2, HistogramEmbedder.dim)
    assert np.allclose(np.linalg.norm(e, axis=1), 1)
    assert preprocess(img, 224).shape == (3, 224, 224)
