#!/usr/bin/env python

import schemdraw
from schemdraw import flow
import paths
import matplotlib as mpl
import matplotlib.pyplot as plt
import global_variables as gv

mpl.rcdefaults()
plt.rcParams['text.usetex'] = gv.use_LaTeX

with schemdraw.Drawing(show=False, fontsize=11) as d:
      d += (initial := flow.Start(w=3, h=1.5).label('SOURCE\nFROM\nCATALOGUE'))
      d += flow.Arrow().down().length(d.unit/3)
      d += (AGN_model := flow.Decision(w=3.2, h=2.4, E='Predicted\nas Galaxy', S='Predicted\nas AGN')
            .label('AGN\nCLASSIFICATION\nMODEL'))
      d += flow.Arrow().length(d.unit/3)
      d += (radio_model := flow.Decision(w=3.5, h=2.4, E='Predicted\nas no radio', S='Predicted\nas radio')
            .label('RADIO\nDETECTION\nMODEL'))
      d += flow.Arrow().length(d.unit/3)
      
      d += (full_z_model := flow.Decision(w=3.2, h=2.4, S='Predicted z')
            .label('REDSHIFT\nPREDICTION\nMODEL'))

      d += (final_line := flow.Arrow().down().at(full_z_model.S).length(d.unit*0.4))
      d += (final_state := flow.StateEnd(r=1.4).label('PREDICTED\nRADIO AGN\nW/REDSHIFT'))
      d += flow.Line().right().at(AGN_model.E).length(d.unit*1.2)
      d += flow.Arrow().down().length(d.unit*0.17)
      d += (discarded := flow.StateEnd(r=1.2).anchor('N').label('DISCARDED\nSOURCE'))
      
      d += flow.Line().right().at(radio_model.E).tox(discarded.S)
      d += flow.Arrow().up().toy(discarded.S)

      d.save(paths.figures / 'flowchart_pipeline_radio_AGN_z.pdf')