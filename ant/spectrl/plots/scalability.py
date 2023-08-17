import os
from spectrl.util.io import save_plot, parse_command_line_options, plot_for_threshold
from matplotlib import pyplot as plt

SPECS = [9, 10, 11, 12, 13]
XS = [2, 4, 8, 12, 16]
THRESHOLDS = [0.8, 0.6, 0.4]
COLORS = ['purple', 'orange', 'deeppink']


flags = parse_command_line_options()
folder = flags['folder']
itno = flags['itno']

folders = [os.path.join(folder, 'spec{}'.format(spec), 'hierarchy') for spec in SPECS]
for i in range(len(THRESHOLDS)):
    plot_for_threshold(itno, folders, XS, THRESHOLDS[i], COLORS[i])

plt.xlabel('# Edges in abstract graph')
plt.ylabel('Average number of samples')

plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0), useMathText=True)
save_plot(folder, 'scalability', True, False)
