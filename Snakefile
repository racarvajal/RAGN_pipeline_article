rule flowchart_HETDEX:
    output:
        "src/tex/figures/flowchart_HETDEX_subsets.pdf"
    script:
        "src/scripts/fig_flowchart_HETDEX.py"

rule flowchart_S82:
    output:
        "src/tex/figures/flowchart_S82_subsets.pdf"
    script:
        "src/scripts/fig_flowchart_Stripe82.py"