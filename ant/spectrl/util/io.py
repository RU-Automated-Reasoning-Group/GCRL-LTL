import os
import cv2
import sys
import getopt
import pickle
import numpy as np

from matplotlib import pyplot as plt
from PIL import Image
from numpy import genfromtxt


def parse_command_line_options(print_options=False):
    optval = getopt.getopt(sys.argv[1:], 'n:d:s:e:a:h:gr', [])
    itno = -1
    folder = ''
    spec_num = 0
    env_num = 0
    rl_algo = 'ars'
    gpu_flag = False
    render = False
    # For Heirarchical ONLY
    num_iter = 100
    for option in optval[0]:
        if option[0] == '-n':
            itno = int(option[1])
        if option[0] == '-d':
            folder = option[1]
        if option[0] == '-s':
            spec_num = int(option[1])
        if option[0] == '-e':
            env_num = int(option[1])
        if option[0] == '-a':
            rl_algo = option[1]
        if option[0] == '-h':
            num_iter = int(option[1])
        if option[0] == '-g':
            gpu_flag = True
        if option[0] == '-r':
            render = True
    flags = {'itno': itno,
             'folder': folder,
             'spec_num': spec_num,
             'env_num': env_num,
             'alg': rl_algo,
             'num_iter': num_iter,
             'gpu_flag': gpu_flag,
             'render': render}
    if print_options:
        print('**** Command Line Options ****')
        for key in flags:
            print('{}: {}'.format(key, flags[key]))
    return flags


def open_log_file(itno, folder):
    '''
    Open a log file to periodically flush data.

    Parameters:
        itno: int
        folder: str
    '''
    fname = _get_prefix(folder) + 'log' + _get_suffix(itno) + '.txt'
    open(fname, 'w').close()
    file = open(fname, 'a')
    return file


def save_object(name, object, itno, folder):
    '''
    Save any pickle-able object.

    Parameters:
        name: str
        object: Object
        itno: int
        folder: str
    '''
    file = open(_get_prefix(folder) + name + _get_suffix(itno) + '.pkl', 'wb')
    pickle.dump(object, file)
    file.close()


def load_object(name, itno, folder):
    '''
    Load pickled object.

    Parameters:
        name: str
        itno: int
        folder: str
    '''
    file = open(_get_prefix(folder) + name + _get_suffix(itno) + '.pkl', 'rb')
    object = pickle.load(file)
    file.close()
    return object


def save_log_info(log_info, itno, folder):
    np.save(_get_prefix(folder) + 'log' + _get_suffix(itno) + '.npy', log_info)


def load_log_info(itno, folder, csv=False):
    if csv:
        return genfromtxt(_get_prefix(folder) + 'log' + _get_suffix(itno) + '.csv', delimiter=',')
    else:
        return np.load(_get_prefix(folder) + 'log' + _get_suffix(itno) + '.npy')


def log_to_file(file, iter, num_transitions, reward, prob, additional_data={}):
    '''
    Log data to file.

    Parameters:
        file: file_handle
        iter: int
        num_transitions: int (number of simulation steps in each iter)
        reward: float
        prob: float (satisfaction probability)
        additional_data: dict
    '''
    file.write('**** Iteration Number {} ****\n'.format(iter))
    file.write('Environment Steps Taken: {}\n'.format(num_transitions))
    file.write('Reward: {}\n'.format(reward))
    file.write('Satisfaction Probability: {}\n'.format(prob))
    for key in additional_data:
        file.write('{}: {}\n'.format(key, additional_data[key]))
    file.write('\n')
    file.flush()


def get_image_dir(itno, folder):
    image_dir = '{}img{}'.format(_get_prefix(folder), _get_suffix(itno))
    if os.path.exists(image_dir) is False:
        os.mkdir(image_dir)
    return image_dir


def generate_video(env, policy, itno, folder, max_step=10000):
    image_dir = get_image_dir(itno, folder)

    done = False
    state = env.reset()
    step = 0
    while not done:
        img_arr = env.render(mode='rgb_array')
        img = Image.fromarray(img_arr)
        img.save(image_dir + '/' + str(step) + '.png')
        action = policy.get_action(state)
        state, _, done, _ = env.step(action)
        step += 1
        if step > max_step:
            done = True

    video_name = image_dir + '/' + 'video.avi'
    images_temp = [img for img in os.listdir(image_dir)]
    images = []
    for i in range(len(images_temp)):
        for j in images_temp:
            directory = str(i) + '.png'
            if directory == j:
                images.append(j)

    frame = cv2.imread(os.path.join(image_dir, images_temp[0]))
    height, width, _ = frame.shape
    video = cv2.VideoWriter(
        video_name, cv2.VideoWriter_fourcc(*'XVID'), 20, (width, height))

    for image in images:
        video.write(cv2.imread(os.path.join(image_dir, image)))

    cv2.destroyAllWindows()
    video.release()


def plot_for_threshold(itno, folders, xs, threshold, color):
    ys = []
    for folder in folders:
        val = 0
        count = 0
        for j in range(itno):
            data = load_log_info(j, folder)
            for pos in range(len(data)):
                if data[pos][-1] >= threshold:
                    val += data[pos][0]
                    count += 1
                    break
        ys.append(val/count)
    plt.subplots_adjust(bottom=0.145, left=0.13)
    plt.rcParams.update({'font.size': 18})
    plt.plot(xs, ys, '-ok', label='z = {}'.format(threshold), color=color)


def plot_error_bar(x, data, color, label, points=False):
    '''
    Plot the error bar from the data.

    Parameters:
        samples_per_iter: int (number of sample rollouts per iteration of the algorithm)
        data: (3+)-tuple of np.array (curve, lower error bar, upper error bar, ...)
        color: color of the plot
        label: string
    '''
    plt.subplots_adjust(bottom=0.126)
    plt.rcParams.update({'font.size': 18})
    if points:
        plt.errorbar(x, data[0], data[0]-data[1], fmt='--o', color=color, label=label)
    else:
        plt.plot(x, data[0], color=color, label=label)
        plt.fill_between(x, data[1], data[2], color=color, alpha=0.15)


def extract_plot_data(folder, column_num, low, up, csv=False):
    '''
    Load and parse log_info to generate error bars

    Parameters:
        folder: string (name of folder)
        column_num: int (column number in log.npy to use)
        l: int (lower limit on run number)
        u: int (upper limit on run number)

    Returns:
        4-tuple of numpy arrays (curve, lower error bar, upper error bar, max_over_runs)
    '''
    log_infos = []
    min_length = 1000000
    for itno in range(low, up):
        log_info = np.transpose(load_log_info(
            itno, folder, csv=csv))[column_num]
        log_info = np.append([0], log_info)
        min_length = min(min_length, len(log_info))
        log_infos.append(log_info)
    log_infos = [log_info[:min_length] for log_info in log_infos]
    data = np.array(log_infos)
    curve = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    max_curve = np.amax(data, axis=0)
    return curve, (curve - std), (curve + std), max_curve


# save and render current plot
def save_plot(folder, name, show=True, scientific=True):
    plt.rcParams.update({'font.size': 14})
    plt.legend()
    ax = plt.gca()
    ax.xaxis.major.formatter._useMathText = True
    if scientific:
        plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
    plt.savefig(_get_prefix(folder) + name + '.pdf', format='pdf')
    if show:
        plt.show()


# get prefix for file name
def _get_prefix(folder):
    if folder == '':
        return ''
    else:
        return folder + '/'


# get suffix from itno
def _get_suffix(itno):
    if itno < 0:
        return ''
    else:
        return str(itno)
