# Network parameter setting
import argparse
def parse_arg():
    parser = argparse.ArgumentParser()

    # basic parameters
    parser.add_argument('--project_name', type=str, default='.model', help='the output path')
    parser.add_argument('--data_path', type=str, default=r'./data/XJTU_SY_dataset.pkl', help='the output path')
    parser.add_argument("--n_epochs", type=int, default=50, help="number of epochs of training")
    parser.add_argument("--batch_size", type=int, default=128, help="size of the batches")
    
    # optimization parameters
    parser.add_argument('--opt', type=str, choices=['sgd', 'adam'], default='sgd', help='the optimizer')
    parser.add_argument("--lr", type=float, default=0.003, help="adam: learning rate")
    parser.add_argument('--momentum', type=float, default=0.9, help='the momentum for sgd')
    parser.add_argument("--weight_decay", type=float, default=1e-5, help="the weight decay during training")
    parser.add_argument("--step", type=int, default=30, help="step of the decay of LR")
    parser.add_argument("--gamma", type=float, default=0.9, help="decay rate of LR")
    parser.add_argument("--use_cuda", type=bool, default=True, help="whether to sue cuda")
    parser.add_argument("--n_works", type=int, default=8, help="workers to load data")

    # Network parameters
    parser.add_argument('--CL', type=list, default=[1, 32, 64, 512], help='channels list of the networks')
    parser.add_argument('--KL', type=list, default=[24, 16, 6], help='kernals list of the networks')  #[16, 8, 4]
    parser.add_argument('--PL', type=list, default=[8,  4,  2], help='paddings list of the networks') #[4,  2, 1]
    parser.add_argument('--SL', type=list, default=[8,  8,  2], help='strides list of the networks')  #[8,  4, 2]
    parser.add_argument("--direct_connect", type=bool, default=True, help="whether to use direct_connect")
    parser.add_argument("--active_type", type=str, default='Soft', help="the type of active function, opt: Soft, Hard, Firm")
    parser.add_argument("--pool_type", type=str, default='Max', help="the type of pooling, opt: Max, Avg")
    parser.add_argument("--n_class", type=int, default=5, help="number of classes")
    parser.add_argument('--unfoldings', type=int, default=4, help='unfoldings of the networks')
    opt = parser.parse_args(args=[])

    return opt

# Ploting parameter setting
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib import rcParams
def plot_parameter(figure_num, figure_size=[6.67, 5], open_frame=True):

    # open_frame --> Whether to draw a semi-open frame diagram

    # font path
    font_path = r".\tools\tnw+simsun.ttf"
    font_manager.fontManager.addfont(font_path)
    prop = font_manager.FontProperties(fname=font_path)

    # font settings
    rcParams['pdf.fonttype'] = 42
    rcParams['font.family'] = 'sans-serif'
    rcParams['font.sans-serif'] = prop.get_name()
    rcParams['font.size'] = 10.5                   # set font size
    rcParams['axes.unicode_minus'] = False

    # generate canvas
    figure = plt.figure(figure_num,figsize=[figure_size[0]/2.54,figure_size[1]/2.54], dpi=300)
    if open_frame:
        ax=plt.gca()
        ax.spines['right'].set_color('none')
        ax.spines['top'].set_color('none')

    # legend location options:
    # 'upper right', 'upper left', 'lower left', 'lower right', 'right', 'center left', 'center right', 'lower center', 'upper center', 'center'

    return figure