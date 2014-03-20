import matplotlib as mpl

def setup_publication_image(fig, height_prop = 1./1.618, single_col = True, replacement_kwargs = None):
    cm_to_inch=0.393701
    if replacement_kwargs == None: replacement_kwargs = {}
    mpl.rcParams['font.size']=8.0
    mpl.rcParams['axes.titlesize']=8.0#'medium'
    mpl.rcParams['xtick.labelsize']=8.0
    mpl.rcParams['ytick.labelsize']=8.0
    mpl.rcParams['lines.markersize']=5.0
    mpl.rcParams['savefig.dpi']=300
    for i in replacement_kwargs.keys():
        mpl.rcParams[i]=replacement_kwargs[i]
    if single_col:
        fig_width = 8.48*cm_to_inch
    else:
        fig_width = 8.48*cm_to_inch*2
    fig.set_figwidth(fig_width)
    fig.set_figheight(fig_width * height_prop)
