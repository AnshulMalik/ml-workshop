def draw_plot(plt, xdata, xlabel, ydata, ylabel, plot_type):
    flag = ''
    if(plot_type == 'dot'):
        flag = 'ro'

    plt.plot(xdata, ydata, flag)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    return plt

def draw_line(plt, start, end):
    plt.plot(start, end, 'k-')

    return plt