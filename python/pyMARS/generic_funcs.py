def setup_publication_image(fig, height_prop = 1./1.618, single_col = True):
    cm_to_inch=0.393701
    import matplotlib as mpl

    mpl.rcParams['font.size']=8.0
    mpl.rcParams['axes.titlesize']=8.0#'medium'
    mpl.rcParams['xtick.labelsize']=8.0
    mpl.rcParams['ytick.labelsize']=8.0
    mpl.rcParams['lines.markersize']=5.0
    mpl.rcParams['savefig.dpi']=300
    if single_col:
        fig_width = 8.48*cm_to_inch
    else:
        fig_width = 8.48*cm_to_inch*2
    fig.set_figwidth(fig_width)
    fig.set_figheight(fig_width * height_prop)
