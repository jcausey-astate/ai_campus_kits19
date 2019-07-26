#!/bin/bash
# Used for seeding initial weights 5 at a time.

set -e

python3 train_2d_multi_view_generator.py --base-dir dicer_data --epochs 8 --view axial --save-last-only --steps=2000 --kt-t-mask-format
cp weights.h5 weights_kt-t_0.h5

python3 train_2d_multi_view_generator.py --base-dir dicer_data --epochs 8 --view axial --save-last-only --steps=2000 --kt-t-mask-format
cp weights.h5 weights_kt-t_1.h5

python3 train_2d_multi_view_generator.py --base-dir dicer_data --epochs 8 --view axial --save-last-only --steps=2000 --kt-t-mask-format
cp weights.h5 weights_kt-t_2.h5

python3 train_2d_multi_view_generator.py --base-dir dicer_data --epochs 8 --view axial --save-last-only --steps=2000 --kt-t-mask-format
cp weights.h5 weights_kt-t_3.h5

python3 train_2d_multi_view_generator.py --base-dir dicer_data --epochs 8 --view axial --save-last-only --steps=2000 --kt-t-mask-format
cp weights.h5 weights_kt-t_4.h5

