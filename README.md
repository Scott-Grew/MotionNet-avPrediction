# MotionNet

A Python program that predicts where cars, bikes and people are about to
go in self-driving car data.

A Waymo car records everything around it: where each road user has been
for the last second, the lane lines, the traffic lights. Given that one
second, this program predicts the next eight — six possible paths per
road user, each with a confidence, because a car rolling up to a
junction might turn or might not, and a single guess would just split
the difference.

{{DATA DERVED IMAGE TO COME}}

*The red car is being predicted. White dashes are where it actually
went. The cyan lines are the six predictions — brighter means more
confident. The model never saw this scene in training.*

## How well it works

minADE is the standard score: how far the closest of the six predictions
is from the truth, on average, in metres. Lower is better. Waymo's own
scoring code produced these numbers on the {{SCENE_COUNT}} held-out
scenes of the validation split:

| minADE (m)  | 3 s | 5 s | 8 s |
|-------------|-----|-----|-----|
| cars        | {{VEH_3S}} | {{VEH_5S}} | {{VEH_8S}} |
| pedestrians | {{PED_3S}} | {{PED_5S}} | {{PED_8S}} |
| cyclists    | {{CYC_3S}} | {{CYC_5S}} | {{CYC_8S}} |

The control, scored through the same code: assume everyone keeps their
current speed and direction. It scores {{CV_8S}} at 8 s. The gap between
that and the table is what the model learned.

These numbers can't be compared to the public leaderboard. Those models
are around twenty times bigger, train on all the data, and score around
1 m at 8 s. This one trains on a quarter of the training split — 122,352
scenes, 16 epochs, two 12-hour sessions on a free Kaggle GPU.

## How it works

1. Waymo's files are unpacked by this repo's own reader and boiled down
   to plain number arrays: one file per scene. The reader is checked
   byte-for-byte against Waymo's, so the data going in is provably the
   data they publish.
2. Before training, the endpoints of every training path are clustered
   into 54 typical destinations per road-user type — straight and far,
   gentle left, hard right, and so on. These are the anchors.
3. A transformer reads the scene and, for each anchor, bends a path
   toward what the scene says. A second head scores how likely each
   anchor is. The loss is the paper's (MultiPath, Chai et al. 2019):
   each training example teaches the path of the anchor its true path
   matches, and teaches the confidence to point at that anchor.
4. At prediction time the 54 are ranked by confidence, near-duplicates
   are dropped, and the best six survive.

Every constant in training was set from a measurement on this model,
not a convention: the gradient clip from the measured size of updates,
the learning rate from a sweep, the anchor rule from counting which
choice groups real paths better. When the confidence head silently
learned to ignore the scene mid-project, an eight-example probe on a
laptop reproduced the failure in minutes and proved the fix before any
GPU time was spent.

## Run it

Tests need no data: `./gate.sh` (45 tests, a few seconds). Training runs
from `train.py`, scoring from `scorer.py`, which drives Waymo's metric
code in a Docker container. Waymo doesn't allow their data to be
shared, so reproducing the table needs a Waymo Open Motion Dataset
account; `stage.py` turns their files into this repo's format.
