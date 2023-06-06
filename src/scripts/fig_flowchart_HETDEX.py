#!/usr/bin/env python

import schemdraw
from schemdraw import flow
import numpy as np
import paths
import matplotlib as mpl
import matplotlib.pyplot as plt
import global_variables as gv

mpl.rcdefaults()
plt.rcParams['text.usetex'] = gv.use_LaTeX

full_size_HTDX       = 15_136_878

labelled_idx         = np.loadtxt(paths.data / 'indices_known.txt')
train_validation_idx = np.loadtxt(paths.data / 'indices_train_validation.txt')
train_idx            = np.loadtxt(paths.data / 'indices_train.txt')
validation_idx       = np.loadtxt(paths.data / 'indices_validation.txt')
calibration_idx      = np.loadtxt(paths.data / 'indices_calibration.txt')
test_idx             = np.loadtxt(paths.data / 'indices_test.txt')

# size_labelled        = len(labelled_idx)  # Sources are incorrect
size_labelled        = 118_734
size_unlabelled      = full_size_HTDX - size_labelled
# size_train_val       = len(train_validation_idx)
size_train_val       = 85_488
size_train_val_cal   = size_train_val + len(calibration_idx)
size_test            = len(test_idx)
size_train           = len(train_idx)
size_val             = len(validation_idx)
size_cal             = len(calibration_idx)
size_val_cal         = size_val + size_cal

with schemdraw.Drawing(show=False, fontsize=14, lw=2.5) as H:
    H += (HETDEX := flow.Terminal(w=3.5, h=1.5).label(f'HETDEX Field\n{full_size_HTDX:,}'.replace(',', '\,')))
    H += schemdraw.elements.lines.Gap().at(HETDEX.S)

    H += (Labelled := flow.RoundBox(w=3.5, h=1.5, anchor='ENE').label(f'Labelled\n{size_labelled:,}'.replace(',', '\,')))
    H += (Unlabelled := flow.RoundBox(w=3.5, h=1.5, anchor='WNW').label(f'Unlabelled\n{size_unlabelled:,}'.replace(',', '\,')))
    H += flow.Arrow().length(H.unit/3).at(HETDEX.S).to(Labelled.N)
    H += flow.Arrow().length(H.unit/3).at(HETDEX.S).to(Unlabelled.N)
    H += schemdraw.elements.lines.Gap().at(Labelled.S)

    H += (Tr_Va_Ca := flow.RoundBox(w=3.5, h=1.5, anchor='ENE').label(f'Train+Validation+\nCalibration\n{size_train_val_cal:,}'.replace(',', '\,')))
    H += (Test := flow.RoundBox(w=3, h=1.5, anchor='WNW').label(f'Test\n{size_test:,}'.replace(',', '\,')))
    H += flow.Arrow().length(H.unit/3).at(Labelled.S).to(Tr_Va_Ca.N)
    H += flow.Arrow().length(H.unit/3).at(Labelled.S).to(Test.N)
    H += schemdraw.elements.lines.Gap().at(Tr_Va_Ca.S)

    H += (Train := flow.RoundBox(w=3.5, h=1.5, anchor='ENE').label(f'Train\n{size_train:,}'.replace(',', '\,')))
    H += (Va_Ca := flow.RoundBox(w=3.5, h=1.5, anchor='WNW').label(f'Validation+\nCalibration\n{size_val_cal:,}'.replace(',', '\,')))
    H += flow.Arrow().length(H.unit/3).at(Tr_Va_Ca.S).to(Train.N)
    H += flow.Arrow().length(H.unit/3).at(Tr_Va_Ca.S).to(Va_Ca.N)
    H += schemdraw.elements.lines.Gap().at(Va_Ca.S)

    H += (Validation := flow.RoundBox(w=3.5, h=1.5, anchor='ENE').label(f'Validation\n{size_val:,}'.replace(',', '\,')))
    H += (Calibration := flow.RoundBox(w=3.5, h=1.5, anchor='WNW').label(f'Calibration\n{size_cal:,}'.replace(',', '\,')))
    H += flow.Arrow().length(H.unit/3).at(Va_Ca.S).to(Validation.N)
    H += flow.Arrow().length(H.unit/3).at(Va_Ca.S).to(Calibration.N)
      
    H.save(paths.figures / 'flowchart_HETDEX_subsets.pdf')
