"""[summary]

Returns:
    [type]: [description]
"""
import matplotlib.pyplot as plt

def draw(x, ys, names=None, title="", x_name="",
         y_name="", file_name="./figures/test.pdf"):
    """[summary]

    Args:
        x ([type]): [description]
        ys ([type]): [description]
        names ([type], optional): [description]. Defaults to None.
        title (str, optional): [description]. Defaults to "".
        x_name (str, optional): [description]. Defaults to "".
        y_name (str, optional): [description]. Defaults to "".
        file_name (str, optional): [description]. Defaults to "./figures/test.pdf".
    """
    if names is None:
        names = ['' for i in range(len(ys))]
    markers = ['o', 'o', 'o', 'o', 'o', 'o', 'o', 'o', 'o', 'o', 'o', 'o', 'o']
    mecs = ['red', 'blue', 'coral', 'darkgreen', 'firebrick', 'orange', 'indigo', 'lavender', 'darkviolet',
            'indianred', 'khaki']
    colors = ['red', 'blue', 'coral', 'darkgreen', 'firebrick', 'orange', 'indigo', 'lavender', 'darkviolet',
              'indianred', 'khaki']
    for i in range(len(ys)):
        plt.plot(
            x,
            ys[i],
            marker=markers[i],
            mec=mecs[i],
            mfc='w',
            color=colors[i],
            label=names[i])
    plt.legend()
    plt.margins(0)
    plt.subplots_adjust(bottom=0.15)
    plt.xlabel(x_name)
    plt.ylabel(y_name)  # Y轴标签
    plt.title(title)  # 标题
    fig1 = plt.gcf()
    plt.show()
    plt.close()
    fig1.savefig(file_name, dpi=100)


# draw_loss
def unpack(pack_y):
    """[summary]

    Args:
        pack_y ([type]): [description]

    Returns:
        [type]: [description]
    """
    ys = [[] for i in range(5)]
    for each in pack_y:
        for i in range(5):
            ys[i].append(each[i])
    return ys
