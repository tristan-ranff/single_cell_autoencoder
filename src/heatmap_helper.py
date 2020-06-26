import matplotlib.pyplot as plt
from matplotlib.transforms import TransformedBbox

def annotate_yranges(groups, ax=None):
    """
    Annotate a group of consecutive yticklabels with a group name.

    Arguments:
    ----------
    groups : dict
        Mapping from group label to an ordered list of group members.
    ax : matplotlib.axes object (default None)
        The axis instance to annotate.
    """
    if ax is None:
        ax = plt.gca()

    label2obj = {ticklabel.get_text() : ticklabel for ticklabel in ax.get_yticklabels()}

    longest = max(groups.keys(), key=len)

    first = groups[longest][0]
    last = groups[longest][-1]

    bbox0 = _get_text_object_bbox(label2obj[first], ax)
    bbox1 = _get_text_object_bbox(label2obj[last], ax)
    largest_dist = min(bbox0.x0, bbox1.x0)

    xl, xh = ax.get_xlim()
    left = xl - (xh - xl)
    Lines = ax.hlines([x*3 for x in range(1, len(groups.keys()))], left, xh, color='k', linewidth=1.4)
    Lines.set_clip_on(False)
    ax.set_xlim((xl, xh))

    for ii, (group, members) in enumerate(groups.items()):
        first = members[0]
        last = members[-1]

        bbox0 = _get_text_object_bbox(label2obj[first], ax)
        bbox1 = _get_text_object_bbox(label2obj[last], ax)


        set_yrange_label(group, bbox0.y0 + bbox0.height/2,
                         bbox1.y0 + bbox1.height/2,
                         largest_dist,
                         -4,
                         ax=ax)


def set_yrange_label(label, ymin, ymax, x, dx=-0.5, ax=None, *args, **kwargs):
    """
    Annotate a y-range.

    Arguments:
    ----------
    label : string
        The label.
    ymin, ymax : float, float
        The y-range in data coordinates.
    x : float
        The x position of the annotation arrow endpoints in data coordinates.
    dx : float (default -0.5)
        The offset from x at which the label is placed.
    ax : matplotlib.axes object (default None)
        The axis instance to annotate.
    """

    if not ax:
        ax = plt.gca()

    dy = ymax - ymin
    ax.annotate(label,
                xy=(x, ymin),
                xytext=(x + dx, ymin + dy/2),
                annotation_clip=False,
                *args, **kwargs,
    )
    ax.annotate(label,
                xy=(x, ymax),
                xytext=(x + dx, ymin + dy/2),
                annotation_clip=False,
                *args, **kwargs,
    )


def _get_text_object_bbox(text_obj, ax):
    # https://stackoverflow.com/a/35419796/2912349
    transform = ax.transData.inverted()
    # the figure needs to have been drawn once, otherwise there is no renderer?
    # plt.ion(); plt.show(); plt.pause(0.001)
    bb = text_obj.get_window_extent(renderer=ax.get_figure().canvas.renderer)
    # handle canvas resizing
    return TransformedBbox(bb, transform)