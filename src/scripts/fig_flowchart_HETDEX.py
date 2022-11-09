#!/usr/bin/env python

import schemdraw
from schemdraw import flow
import paths
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rcdefaults()
plt.rcParams['text.usetex'] = True

with schemdraw.Drawing(show=False, fontsize=14, lw=2.5) as H:
    H += (HETDEX := flow.Terminal(w=3.5, h=1.5).label('HETDEX Field\n6 729 647'))
    H += schemdraw.elements.lines.Gap().at(HETDEX.S)

    H += (Labelled := flow.RoundBox(w=3, h=1.5, anchor='ENE').label('Labelled\n83 409'))
    H += (Unlabelled := flow.RoundBox(w=3, h=1.5, anchor='WNW').label('Unlabelled\n6 646 238'))
    H += flow.Arrow().length(H.unit/3).at(HETDEX.S).to(Labelled.N)
    H += flow.Arrow().length(H.unit/3).at(HETDEX.S).to(Unlabelled.N)
    H += schemdraw.elements.lines.Gap().at(Labelled.S)

    H += (Tr_Te_Ca := flow.RoundBox(w=3, h=1.5, anchor='ENE').label('Train+Test+\nCalibration\n66 727'))
    H += (Validation := flow.RoundBox(w=3, h=1.5, anchor='WNW').label('Validation\n16 682'))
    H += flow.Arrow().length(H.unit/3).at(Labelled.S).to(Tr_Te_Ca.N)
    H += flow.Arrow().length(H.unit/3).at(Labelled.S).to(Validation.N)
    H += schemdraw.elements.lines.Gap().at(Tr_Te_Ca.S)

    H += (Train := flow.RoundBox(w=3, h=1.5, anchor='ENE').label('Train\n53 381'))
    H += (Te_Ca := flow.RoundBox(w=3, h=1.5, anchor='WNW').label('Test+\nCalibration\n13 346'))
    H += flow.Arrow().length(H.unit/3).at(Tr_Te_Ca.S).to(Train.N)
    H += flow.Arrow().length(H.unit/3).at(Tr_Te_Ca.S).to(Te_Ca.N)
    H += schemdraw.elements.lines.Gap().at(Te_Ca.S)

    H += (Test := flow.RoundBox(w=3, h=1.5, anchor='ENE').label('Test\n6 673'))
    H += (Calibration := flow.RoundBox(w=3, h=1.5, anchor='WNW').label('Calibration\n6 673'))
    H += flow.Arrow().length(H.unit/3).at(Te_Ca.S).to(Test.N)
    H += flow.Arrow().length(H.unit/3).at(Te_Ca.S).to(Calibration.N)
      
    H.save(paths.figures / 'flowchart_HETDEX_subsets.pdf')
# 
# with schemdraw.Drawing(show=False, fontsize=14, lw=2.5) as S:
#     S += (HETDEX := flow.Terminal(w=3, h=1.5).label('Stripe 82\n369,093'))
#     S += schemdraw.elements.lines.Gap().at(HETDEX.S)
# 
#     S += (Labelled := flow.RoundBox(w=3, h=1.5, anchor='ENE').label('Labelled\n3,304'))
#     S += (Unlabelled := flow.RoundBox(w=3, h=1.5, anchor='WNW').label('Unlabelled\n365,789'))
#     S += flow.Arrow().length(S.unit/3).at(HETDEX.S).to(Labelled.N)
#     S += flow.Arrow().length(S.unit/3).at(HETDEX.S).to(Unlabelled.N)
# 
#     S.save(paths.figures / 'flowchart_S82_subsets.pdf')