import numpy as np
import pandas as pd
import math
from scipy.fftpack import fft
import matplotlib.pyplot as plt
import re
import os

root = os.path.dirname(os.path.abspath('__file__'))
import sys

sys.path.append(root)
from code_1.utils.skopt_plots import plot_convergence, plot_objective, plot_evaluations
from code_1.utils.fit_line import compute_multi_linear_fit

# plt.rcParams['figure.figsize']=(10,8)
plt.rcParams['font.size'] = 9
import matplotlib
matplotlib.rcParams['axes.unicode_minus'] =False

# plt.rcParams["figure.figsize"] = [7.48, 5.61]
# plt.rcParams['image.cmap']='plasma'
# plt.rcParams['axes.linewidth']=2

def plot_cv_error(data_path, labels, mode='avg'):
    # logger.info('Plot cross validation MSE...')
    # logger.info('Data path:{}'.format(data_path))
    # logger.info('Labels:{}'.format(labels))
    if isinstance(data_path, str):
        data_path = [data_path]
        labels = [labels]
    plt.figure(figsize=(7.48, 7.48))
    plt.xlabel('CV')
    plt.ylabel('MSE')
    for path, label in zip(data_path, labels):
        # logger.info('Read cv results of {}'.format(path))
        dev_cv = {}
        test_cv = {}
        for file_ in os.listdir(path):
            if '.csv' in file_ and 'seed' not in file_ and 'optimized_params' not in file_:
                # logger.info('cv-file:{}'.format(file_))
                cv = int(re.findall(r"(?<=cv)\d+", file_)[0])
                # logger.info('cv={}'.format(cv))
                data = pd.read_csv(path + file_)
                dev_metrics = data['dev_nrmse'][0]
                test_metrics = data['test_nrmse'][0]
                # logger.info('Development metrics={}'.format(dev_metrics))
                dev_cv[cv] = dev_metrics
                test_cv[cv] = test_metrics
        # logger.debug('Development cv dict before sort:{}'.format(dev_cv))
        # logger.debug('Testing cv dict before sort:{}'.format(test_cv))
        dev_cv = dict(sorted(dev_cv.items()))
        test_cv = dict(sorted(test_cv.items()))
        # logger.info('Cross validation development dict:{}'.format(dev_cv))
        # logger.info('Cross validation folds:{}'.format(dev_cv.keys()))
        # logger.info('Cross validation MSE:{}'.format(dev_cv.values()))
        plt.plot(list(dev_cv.keys()), list(dev_cv.values()), marker='o', label=label + 'dev')
        plt.plot(list(test_cv.keys()), list(test_cv.values()), marker='o', label=label + 'test')
        plt.legend()
    plt.tight_layout()
    # plt.show()


def plot_pacf(station, decomposer=None, wavelet_level=None, measurement_time='month', measurement_unit="$10^8m^3$", ):
    if decomposer == None:
        file_path = root + '/' + station + '/data/PACF.csv'
        save_path = root + '/' + station + '/graph/'
    elif wavelet_level == None:
        file_path = root + '/' + station + '_' + decomposer + '/data/PACF.csv'
        save_path = root + '/' + station + '_' + decomposer + '/graph/'
    elif decomposer == 'modwt':
        file_path = root + '/' + station + '_' + decomposer + '/data-tsdp/' + wavelet_level + '/PACF.csv'
        save_path = root + '/' + station + '_' + decomposer + '/graph/'
    else:
        file_path = root + '/' + station + '_' + decomposer + '/data/' + wavelet_level + '/PACF.csv'
        save_path = root + '/' + station + '_' + decomposer + '/graph/'
    data = pd.read_csv(file_path)
    up = data.pop('UP')
    low = data.pop('LOW')
    n_pacf = data.shape[1]
    lags = list(range(0, data.shape[0]))
    t = list(range(-1, data.shape[0]))
    z_line = np.zeros(len(t))
    signals = data.columns.tolist()
    height = {
        1: 3,
        2: 3,
        3: 6,
        4: 6,
        5: 9,
        6: 9,
        7: 12,
        8: 12,
        9: 15,
        10: 15,
        11: 18,
        12: 18,
        13: 21,
        14: 21,
    }
    plt.figure(figsize=(7.48, height[n_pacf]))
    for i in range(n_pacf):
        if n_pacf > 1:
            plt.subplot(math.ceil(n_pacf / 2), 2, i + 1)
        plt.title(signals[i], loc='left', )
        plt.bar(lags, data[signals[i]], color='b', width=0.8)
        plt.plot(lags, up, '--', color='r')
        plt.plot(lags, low, '--', color='r')
        plt.plot(t, z_line, '--', color='b', linewidth=0.5)
        plt.xlim(-1, 20)
        plt.ylim(-1, 1)
        plt.xticks([0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20], )
        plt.yticks()
        plt.xlabel('Lag(' + measurement_time + ')', )
        plt.ylabel('PACF', )
    plt.tight_layout()

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    if wavelet_level != None:
        plt.savefig(save_path + 'PACF(' + wavelet_level + ').png', format='PNG', dpi=300)
    else:
        plt.savefig(save_path + 'PACF.png', format='PNG', dpi=300)
    # plt.show()


def plot_rela_pred(records, predictions, fig_savepath, time=None, measurement_time='month',
                   measurement_unit="$10^8m^3$",
                   figsize=(7.48, 7), format='PNG', dpi=300):
    """ 
    Plot the relations between the records and predictions.
    Args:
        records: the actual measured records.
        predictions: the predictions obtained by model
        fig_savepath: the path where the plot figure will be saved.
    """
    # logger.info('Plot predictions and correlations...')
    if isinstance(records, pd.DataFrame) or isinstance(records, pd.Series):
        records = records.values
    elif isinstance(predictions, pd.DataFrame) or isinstance(predictions, pd.Series):
        predictions = predictions.values
    length = records.size
    if time is None:
        time = np.linspace(start=1, stop=length, num=length)
    plt.figure(figsize=figsize)
    # ax1 = plt.subplot2grid((1,5), (0,0), colspan=3)
    # ax2 = plt.subplot2grid((1,5), (0,3),colspan=2,aspect='equal')
    ax1 = plt.subplot2grid((2, 2), (0, 0), colspan=2)
    ax2 = plt.subplot2grid((2, 2), (1, 0), colspan=1, aspect='equal')
    ax3 = plt.subplot2grid((2, 2), (1, 1), colspan=1, )

    # ax1.set_xticks([])
    # ax1.set_yticks([])
    ax1.set_xlabel('year(' + measurement_time + ')', )
    ax1.set_ylabel(r'population(' + measurement_unit + ')', )
    ax1.plot(time, records, '-', color='blue', label='Records', linewidth=1.0)
    ax1.plot(time, predictions, '--', color='red', label='Predictions', linewidth=1.0)
    ax1.legend(
        # loc='upper left',
        loc=0,
        # bbox_to_anchor=(0.005,1.2),
        shadow=False,
        frameon=False,
    )
    # logger.info('records=\n{}'.format(records))
    # logger.info('predictions=\n{}'.format(predictions))

    pred_min = predictions.min()
    pred_max = predictions.max()
    record_min = records.min()
    record_max = records.max()
    if pred_min < record_min:
        xymin = pred_min
    else:
        xymin = record_min
    if pred_max > record_max:
        xymax = pred_max
    else:
        xymax = record_max

    # logger.info('xymin={}'.format(xymin))
    # logger.info('xymax={}'.format(xymax))

    xx = np.arange(start=xymin, stop=xymax + 1, step=1.0)
    coeff = np.polyfit(predictions, records, 1)
    linear_fit = coeff[0] * xx + coeff[1]
    # print('a:{}'.format(coeff[0]))
    # print('b:{}'.format(coeff[1]))
    # ax2.set_xticks()
    # ax2.set_yticks()
    ax2.set_xlabel(r'Predictions(' + measurement_unit + ')', )
    ax2.set_ylabel(r'Records(' + measurement_unit + ')', )
    # ax2.plot(predictions, records, 'o', color='blue', label='',markersize=6.5)
    ax2.plot(predictions, records, 'o', markerfacecolor='w', markeredgecolor='blue', markersize=6.5)
    # ax2.plot(predictions, linear_fit, '--', color='red', label='Linear fit',linewidth=1.0)
    # ax2.plot(predictions, ideal_fit, '-', color='black', label='Ideal fit',linewidth=1.0)
    ax2.plot(xx, linear_fit, '--', color='red', label='Linear fit', linewidth=1.0)
    ax2.plot([xymin, xymax], [xymin, xymax], '-', color='black', label='Ideal fit', linewidth=1.0)
    ax2.set_xlim([xymin, xymax])
    ax2.set_ylim([xymin, xymax])
    ax2.legend(
        # loc='upper left',
        loc=0,
        # bbox_to_anchor=(0.05,1),
        shadow=False,
        frameon=False,
    )
    error = pd.DataFrame(records - predictions)
    ax3.set_xlabel('Error', )
    ax3.set_ylabel('Density', )
    error.plot.kde(color='red', linewidth=1, ax=ax3)
    ax3.legend(
        # loc='upper left',
        loc=0,
        # bbox_to_anchor=(0.05,1),
        shadow=False,
        frameon=False,
    )

    # plt.subplots_adjust(left=0.08, bottom=0.12, right=0.98, top=0.98, hspace=0.1, wspace=0.2)
    plt.tight_layout()
    plt.savefig(fig_savepath, format=format, dpi=dpi)
    # plt.show()


def plot_rela_pred_2(records, predictions0, predictions, fig_savepath, measurement_time='month',
                     measurement_unit="$10^8m^3$", figsize=(7.48, 7), format='PNG', dpi=300):
    # logger.info('Plot predictions and correlations...')
    if isinstance(records, pd.DataFrame) or isinstance(records, pd.Series):
        records = records.values
    elif isinstance(predictions0, pd.DataFrame) or isinstance(predictions0, pd.Series):
        predictions0 = predictions0.values
    elif isinstance(predictions, pd.DataFrame) or isinstance(predictions, pd.Series):
        predictions = predictions.values

    length = len(records)
    t = np.linspace(start=1, stop=length, num=length)
    plt.figure(figsize=figsize)
    ax1 = plt.subplot2grid((2, 2), (0, 0), colspan=2)
    ax2 = plt.subplot2grid((2, 2), (1, 0), colspan=1, aspect='equal')
    ax3 = plt.subplot2grid((2, 2), (1, 1), colspan=1, )
    # ax1 = plt.subplot2grid((8,1), (0,0), rowspan=2)
    # ax2 = plt.subplot2grid((8,1), (2,0),rowspan=6,aspect='equal')

    # ax1.set_xticks([])
    # ax1.set_yticks([])
    ax1.set_xlabel('Time(' + measurement_time + ')', )
    ax1.set_ylabel(r'Streamflow(' + measurement_unit + ')', )
    ax1.plot(t, records, '-', color='blue', label='Records', linewidth=1.0)
    ax1.plot(t, predictions0, '--', color='purple', label='Predictions', linewidth=1.0, alpha=0.6)
    ax1.plot(t, predictions, '--', color='red', label='Retrain predictions', linewidth=1.0)
    ax1.legend(
        # loc='upper left',
        loc=0,
        # bbox_to_anchor=(0.005,1.2),
        shadow=False,
        frameon=False,
    )
    # logger.info('records=\n{}'.format(records))
    # logger.info('predictions=\n{}'.format(predictions))

    recordss = {'s0': records, 's': records}
    predss = {'s0': predictions0, 's': predictions}

    xx, linear_list, xymin, xymax = compute_multi_linear_fit(recordss, predss)
    xx = np.arange(start=xymin, stop=xymax + 1, step=1.0)
    coeff = np.polyfit(predictions, records, 1)
    linear_fit = coeff[0] * xx + coeff[1]
    # print('a:{}'.format(coeff[0]))
    # print('b:{}'.format(coeff[1]))
    # ax2.set_xticks()
    # ax2.set_yticks()
    ax2.set_xlabel(r'Predictions(' + measurement_unit + ')', )
    ax2.set_ylabel(r'Records(' + measurement_unit + ')', )
    ax2.plot(predictions0, records, 'o', label='Preds VS. Records', markerfacecolor='w', markeredgecolor='purple',
             markersize=6.5, alpha=0.6)
    ax2.plot(predictions, records, 'o', label='Retrain preds VS. Records', markerfacecolor='w', markeredgecolor='blue',
             markersize=6.5)
    ax2.plot(xx, linear_list['s0'], '--', color='purple', label='Preds linear fit', linewidth=1.0)
    ax2.plot(xx, linear_list['s'], '--', color='red', label='Retrain preds linear fit', linewidth=1.0)
    ax2.plot([xymin, xymax], [xymin, xymax], '-', color='black', label='Ideal fit', linewidth=1.0)
    ax2.set_xlim([xymin, xymax])
    ax2.set_ylim([xymin, xymax])
    ax2.legend(
        # loc='upper left',
        loc=0,
        # bbox_to_anchor=(0.05,1),
        shadow=False,
        frameon=False,
    )
    error0 = pd.DataFrame(records - predictions0, columns=['Error for predictions'])
    error = pd.DataFrame(records - predictions, columns=['Error for retrain predictions'])
    ax3.set_xlabel('Error', )
    ax3.set_ylabel('Density', )
    error0.plot.kde(color='purple', linewidth=1, ax=ax3)
    error.plot.kde(color='red', linewidth=1, ax=ax3)
    ax3.legend(
        # loc='upper left',
        loc=0,
        # bbox_to_anchor=(0.05,1),
        shadow=False,
        frameon=False,
    )

    # plt.subplots_adjust(left=0.08, bottom=0.12, right=0.98, top=0.98, hspace=0.1, wspace=0.2)
    plt.tight_layout()
    plt.savefig(fig_savepath, format=format, dpi=dpi)
    # plt.show()


def plot_history(history, path1, path2, figsize=(7.48, 5.61), format='PNG', dpi=300):
    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch

    plt.figure(figsize=figsize)
    plt.xlabel('Epoch')
    plt.ylabel('Mean abs error')
    plt.plot(hist['epoch'], hist['mean_absolute_error'], label='Train error')
    plt.plot(hist['epoch'], hist['val_mean_absolute_error'], label='Val error')
    # plt.ylim([0,0.04])
    plt.legend()
    plt.savefig(path1, format=format, dpi=dpi)

    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Mean Square Error')
    plt.plot(hist['epoch'], hist['mean_squared_error'], label='Train Error')
    plt.plot(hist['epoch'], hist['val_mean_squared_error'], label='Val Error')
    # plt.ylim([0,0.04])
    plt.legend()
    plt.tight_layout()
    plt.savefig(path2, format=format, dpi=dpi)
    # plt.show()


def plot_error_distribution(records, predictions, fig_savepath, figsize=(3.54, 2.0), format='PNG', dpi=300):
    """
    Plot error distribution from the predictions and labels.
    Args:
        predictions: prdictions obtained by ml 
        records: real records
        fig_savepath: path to save a error distribution figure

    Return
        A figure of error distribution
    """
    plt.figure(figsize=figsize)
    error = predictions - records
    # plt.hist(error,bins=25)
    # plt.hist(error, 50, density=True,log=True, facecolor='g', alpha=0.75)
    plt.hist(error, 20, density=True, log=True, )
    plt.xlabel('Prediction Error')
    plt.ylabel('count')
    plt.tight_layout()
    plt.savefig(fig_savepath, format=format, dpi=dpi)
    # plt.show()


def plot_convergence_(OptimizeResult, fig_savepath, figsize=(5.51, 3.54), format='PNG', dpi=300):  # (5.51,3.54)
    """
    Plot one or several convergence traces.
    Parameters
    args[i] [OptimizeResult, list of OptimizeResult, or tuple]: The result(s) for which to plot the convergence trace.
    
    if OptimizeResult, then draw the corresponding single trace;
    if list of OptimizeResult, then draw the corresponding convergence traces in transparency, along with the average convergence trace;
    if tuple, then args[i][0] should be a string label and args[i][1] an OptimizeResult or a list of OptimizeResult.
    ax [Axes, optional]: The matplotlib axes on which to draw the plot, or None to create a new one.
    
    true_minimum [float, optional]: The true minimum value of the function, if known.
    
    yscale [None or string, optional]: The scale for the y-axis.

    fig_savepath [string]: The path to save a convergence figure
    
    Returns
    ax: [Axes]: The matplotlib axes.
    """
    fig = plt.figure(figsize=figsize)
    ax0 = fig.add_subplot(111)
    plot_convergence(OptimizeResult, ax=ax0, true_minimum=0.0, )
    plt.title("")
    ax0.set_ylabel(r"minimum MSE after $n$ calls")
    # plt.tight_layout()
    plt.subplots_adjust(left=0.12, bottom=0.12, right=0.96, top=0.94, hspace=0.1, wspace=0.2)
    plt.savefig(fig_savepath, format=format, dpi=dpi)
    # plt.show()


def plot_evaluations_(OptimizeResult, dimensions, fig_savepath, figsize=(7.48, 7.48), format='PNG', dpi=300):
    """
    Visualize the order in which points where sampled.

    The scatter plot matrix shows at which points in the search space and in which order samples were evaluated. Pairwise scatter plots are shown on the off-diagonal for each dimension of the search space. The order in which samples were evaluated is encoded in each point's color. The diagonal shows a histogram of sampled values for each dimension. A red point indicates the found minimum.

    Note: search spaces that contain Categorical dimensions are currently not supported by this function.

    Parameters
    result [OptimizeResult] The result for which to create the scatter plot matrix.

    bins [int, bins=20]: Number of bins to use for histograms on the diagonal.

    dimensions [list of str, default=None] Labels of the dimension variables. None defaults to space.dimensions[i].name, or if also None to ['X_0', 'X_1', ..].

    fig_savepath [string]: The path to save a convergence figure

    Returns
    ax: [Axes]: The matplotlib axes.
    """
    plot_evaluations(OptimizeResult, figsize=figsize, dimensions=dimensions)
    plt.tight_layout()
    # plt.subplots_adjust(left=0.08, bottom=0.12, right=0.98, top=0.98, hspace=0.1, wspace=0.2)
    plt.savefig(fig_savepath, format=format, dpi=dpi)
    # plt.show()


def plot_objective_(OptimizeResult, dimensions, fig_savepath, figsize=(7.48, 7.48), format='PNG', dpi=300):
    """
    Pairwise partial dependence plot of the objective function.

    The diagonal shows the partial dependence for dimension i with respect to the objective function. The off-diagonal shows the partial dependence for dimensions i and j with respect to the objective function. The objective function is approximated by result.model.
    
    Pairwise scatter plots of the points at which the objective function was directly evaluated are shown on the off-diagonal. A red point indicates the found minimum.
    
    Note: search spaces that contain Categorical dimensions are currently not supported by this function.
    
    Parameters
    result [OptimizeResult] The result for which to create the scatter plot matrix.
    
    levels [int, default=10] Number of levels to draw on the contour plot, passed directly to plt.contour().
    
    n_points [int, default=40] Number of points at which to evaluate the partial dependence along each dimension.
    
    n_samples [int, default=250] Number of random samples to use for averaging the model function at each of the n_points.
    
    size [float, default=2] Height (in inches) of each facet.
    
    zscale [str, default='linear'] Scale to use for the z axis of the contour plots. Either 'linear' or 'log'.
    
    dimensions [list of str, default=None] Labels of the dimension variables. None defaults to space.dimensions[i].name, or if also None to ['X_0', 'X_1', ..].

    fig_savepath [string]: The path to save a convergence figure
    
    Returns
    ax: [Axes]: The matplotlib axes.
    """
    plot_objective(OptimizeResult, figsize=figsize, dimensions=dimensions)
    plt.tight_layout()
    # plt.subplots_adjust(left=0.08, bottom=0.12, right=0.98, top=0.98, hspace=0.1, wspace=0.2)
    plt.savefig(fig_savepath, format=format, dpi=dpi)
    # plt.show()


def plot_subsignals_pred(
        predictions, records,
        test_len, full_len,
        fig_savepath,
        subsignals_name=None,
        measurement_time='month', measurement_unit="$10^8m^3$",
        figsize=(7.48, 7.48), format='PNG', dpi=300, ):
    assert predictions.shape[1] == records.shape[1]

    if subsignals_name == None:
        subsignals_name = []
        for k in range(1, predictions.shape[1] + 1):
            subsignals_name.append('S' + str(k))

    plt.figure(figsize=figsize)
    t = range(full_len - test_len + 1, full_len + 1)
    for i in range(1, predictions.shape[1] + 1):
        plt.subplot(math.ceil(predictions.shape[1] / 2.0), 2, i)
        plt.title(subsignals_name[i - 1], loc='left')
        plt.xticks()
        plt.yticks()
        if i == predictions.shape[1] or i == predictions.shape[1] - 1:
            plt.xlabel('Time(' + measurement_time + ')', )
        plt.ylabel(r"Streamflow(" + measurement_unit + ")", )
        if i == 1:
            plt.plot(t, records.iloc[:, i - 1], '-', color='blue', label='records', linewidth=1.0)
            plt.plot(t, predictions.iloc[:, i - 1], '--', color='red', label='', linewidth=1.0)
        if i == 2:
            plt.plot(t, records.iloc[:, i - 1], '-', color='blue', label='', linewidth=1.0)
            plt.plot(t, predictions.iloc[:, i - 1], '--', color='red', label='predictions', linewidth=1.0)
        plt.plot(t, records.iloc[:, i - 1], '-', color='blue', label='', linewidth=1.0)
        plt.plot(t, predictions.iloc[:, i - 1], '--', color='red', label='', linewidth=1.0)
        plt.legend(
            loc=3,
            bbox_to_anchor=(0.3, 1.02, 1, 0.102),
            # bbox_transform=plt.gcf().transFigure,
            ncol=2,
            shadow=False,
            frameon=False,
        )
    plt.tight_layout()
    # plt.subplots_adjust(left=0.1, bottom=0.08, right=0.96,top=0.94, hspace=0.6, wspace=0.3)
    plt.savefig(fig_savepath, transparent=False, format=format, dpi=dpi)


def plot_decompositions(signal, figsize=None, save_path=None, measurement_time='month', measurement_unit="$10^8m^3$", ):
    cols = signal.columns.values
    # logger.info('cols={}'.format(cols))
    T = signal.shape[0]
    t = np.arange(start=1, stop=T + 1, step=1, dtype=np.float) / T
    freqs = t - 0.5 - 1 / T
    if figsize == None:
        figsize = (7.48, 1 * len(cols))
    plt.figure(figsize=figsize)
    for i in range(len(cols)):
        subsignal = signal[cols[i]].values
        plt.subplot(len(cols), 2, 2 * i + 1)
        plt.title(cols[i])
        plt.plot(subsignal, c='b')
        if i == len(cols) - 1:
            plt.xlabel('Time(' + measurement_time + ')')
        else:
            plt.xticks([])
        plt.ylabel(r"Streamflow(" + measurement_unit + ")", )
        plt.subplot(len(cols), 2, 2 * i + 2)
        plt.title(cols[i])
        plt.plot(freqs, abs(fft(subsignal)), c='b', lw=0.8, zorder=0)
        if i == len(cols) - 1:
            plt.xlabel('Frequency(1/' + measurement_time + ')')
        else:
            plt.xticks([])
        plt.ylabel('Amplitude')
    plt.tight_layout()
    # plt.show()
