import os
from matplotlib import pyplot as plt
from spectrl.util.io import (plot_error_bar, extract_plot_data, save_plot,
                             parse_command_line_options)

# mypy: ignore-errors

'''
Comment out baselines for which results are not available to obtain
partial plots.
'''

'''
0: Samples
1: Time
-1: Probability
'''

x_index = 0
start_itno = 0
x_max = 1e9

flags = parse_command_line_options()
folder_n = flags['folder']
itno = flags['itno']
spec = flags['spec_num']
folder = os.path.join(folder_n, 'spec{}'.format(spec))

plot_name = "spec"+str(spec)

hierarchy_folder = os.path.join(folder, "hierarchy")
spectrl_folder = os.path.join(folder, "spectrl")
tltl_folder = os.path.join(folder, "tltl")


xh, _, _, _ = extract_plot_data(hierarchy_folder, x_index, start_itno, itno)
yh = extract_plot_data(hierarchy_folder, -1, start_itno, itno)
plot_error_bar(xh, yh, 'blue', 'DiRL (Ours)', points=True)
x_max = min(x_max, xh[-1]+3000)


xs, _, _, _ = extract_plot_data(spectrl_folder, x_index, start_itno, itno)
ys = extract_plot_data(spectrl_folder, -1, start_itno, itno)
plot_error_bar(xs, ys, 'seagreen', 'SPECTRL')
x_max = min(x_max, xs[-1])

xt, _, _, _ = extract_plot_data(tltl_folder, x_index, start_itno, itno)
yt = extract_plot_data(tltl_folder, -1, start_itno, itno)
plot_error_bar(xt, yt, 'tomato', 'TLTL')
x_max = min(x_max, xt[-1])

plt.xlim(right=x_max, left=-0.01)
save_plot(folder_n, plot_name, False)
