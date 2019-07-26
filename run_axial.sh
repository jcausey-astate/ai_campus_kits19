#!/bin/bash
# Used for seeding initial weights 5 at a time.

set -e

python3 train_2d_multi_view_generator.py --epochs 8 --view axial --save-last-only --steps=1000
cp weights.h5 weights_0.h5

python3 train_2d_multi_view_generator.py --epochs 8 --view axial --save-last-only --steps=1000
cp weights.h5 weights_1.h5

python3 train_2d_multi_view_generator.py --epochs 8 --view axial --save-last-only --steps=1000
cp weights.h5 weights_2.h5

python3 train_2d_multi_view_generator.py --epochs 8 --view axial --save-last-only --steps=1000
cp weights.h5 weights_3.h5

python3 train_2d_multi_view_generator.py --epochs 8 --view axial --save-last-only --steps=1000
cp weights.h5 weights_4.h5
