import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.ticker import MaxNLocator
import matplotlib.colors as colors
import matplotlib.cm as cmx
import matplotlib.pyplot as pt

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

def setup_axis_publication(ax, n_xticks = None, n_yticks = None):
    if n_yticks!= None: ax.yaxis.set_major_locator(MaxNLocator(n_yticks))
    if n_xticks!= None: ax.xaxis.set_major_locator(MaxNLocator(n_xticks))

def cbar_ticks(cbar_ax, n_ticks = 5):
    cbar_ax.locator = MaxNLocator(n_ticks)

def create_cbar_ax(original_ax, pad = 3, loc = "right", prop = 5):
    divider = make_axes_locatable(original_ax)
    return divider.append_axes(loc, "{}%".format(prop), pad="{}%".format(pad))



def new_color_cycle(min_val, max_val,cmap='jet',):
    '''This creates a cycle through a colormap
    SRH: 27Apr2015
    '''
    cmap = cmap
    jet = cm = pt.get_cmap(cmap)
    cNorm  = colors.Normalize(vmin=min_val, vmax=max_val)
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=jet)
    return scalarMap.to_rgba

def make_figure_share_xy_row(nrows, ncols):
    fig = pt.figure()
    ax_overall = []
    #fig, ax = pt.subplots(nrows = 3, ncols = ncols, sharex = True)
    for j in range(ncols):
        ax_tmp = []
        for i in range(nrows):
            if j==0:
                ax_tmp.append(fig.add_subplot(nrows,ncols,i*ncols + j + 1))
            else:
                ax_tmp.append(fig.add_subplot(nrows,ncols,i*ncols + j + 1, sharex = ax_overall[0][i], sharey = ax_overall[0][i]))
        ax_overall.append(ax_tmp)
    return fig, ax_overall
