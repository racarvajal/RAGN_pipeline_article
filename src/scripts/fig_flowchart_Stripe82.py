#!/usr/bin/env python

import schemdraw
from schemdraw import flow
import paths

with schemdraw.Drawing(show=False, fontsize=14, lw=2.5) as S:
    S += (HETDEX := flow.Terminal(w=3, h=1.5).label('Stripe 82\n369,093'))
    S += schemdraw.elements.lines.Gap().at(HETDEX.S)

    S += (Labelled := flow.RoundBox(w=3, h=1.5, anchor='ENE').label('Labelled\n3,304'))
    S += (Unlabelled := flow.RoundBox(w=3, h=1.5, anchor='WNW').label('Unlabelled\n365,789'))
    S += flow.Arrow().length(S.unit/3).at(HETDEX.S).to(Labelled.N)
    S += flow.Arrow().length(S.unit/3).at(HETDEX.S).to(Unlabelled.N)

    S.save(paths.figures / 'flowchart_S82_subsets.pdf')