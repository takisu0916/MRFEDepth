from __future__ import absolute_import, division, print_function
import os
import argparse
import sys

project_dir = os.path.dirname(__file__)


class MRFEDepthOptions:
    def __init__(self):
        self.parser = argparse.ArgumentParser(description='MRFEDepth options')


        self.parser.add_argument('--data_path',
                                 # default=r'F:\Datasets',
                                 # default=r'/data/wrj_myself0166_0506_smallResolution',
                                 default=r'/hy-tmp/wrj_myself0166_0506_smallResolution',
                                 type=str,
                                 help='训练数据存放的路径')
        self.parser.add_argument('--log_path',
                                 default=os.path.join(project_dir, 'logs'),
                                 type=str,
                                 help='训练日志存放的路径')


        self.parser.add_argument('--model_name',
                                 default='model_D18P18F18_eia_k5_dl',
                                 type=str,
                                 help='模型保存的文件夹')
        self.parser.add_argument('--split',
                                 default='eigen_zhou',
                                 choices=["eigen_zhou", "eigen_full", "odom", "benchmark", "cityscapes_preprocessed"],
                                 type=str,
                                 help='训练时选择的数据集分割方法')
        self.parser.add_argument('--num_layers',
                                 default=18,
                                 choices=[18, 34, 50, 101, 152],
                                 type=int,
                                 help="使用resnet的层数")
        self.parser.add_argument('--dataset',
                                 default='kitti',
                                 choices=["kitti", "kitti_odom", "kitti_depth", "kitti_test",
                                          "cityscapes_preprocessed"],
                                 type=str,
                                 help='训练使用的数据集名称')
        self.parser.add_argument('--height',
                                 default=288,  # 9
                                 type=int,
                                 help='输入图像的高',
                                 )
        self.parser.add_argument('--width',
                                 default=512,  # 16
                                 type=int,
                                 help='输入图像的宽',
                                 )
        self.parser.add_argument('--disparity_smoothness',
                                 default=1e-3,
                                 type=float,
                                 help='视差平滑权重')
        self.parser.add_argument('--scales',
                                 default=[0, 1, 2, 3],
                                 nargs='+',
                                 type=int,
                                 help='多尺度求loss')
        self.parser.add_argument('--frame_ids',
                                 default=[0, -1, 1],
                                 nargs='+',
                                 type=int,
                                 help='帧编号')
        self.parser.add_argument('--min_depth',
                                 default=0.1,
                                 type=float,
                                 help='最小的深度')
        self.parser.add_argument('--max_depth',
                                 default=100.0,
                                 type=float,
                                 help='最大的深度')
        self.parser.add_argument('--png',
                                 default=False,
                                 action='store_true',
                                 help='(可选项)若设置就使用png格式的图片')
        self.parser.add_argument('--use_stereo',
                                 default=False,
                                 action='store_true',
                                 help='(可选项)若设置就使用双目图像训练')

        self.parser.add_argument('--batch_size',
                                 default=8,
                                 type=int,
                                 help='输入网络每批次的图像数量')
        self.parser.add_argument('--learning_rate',
                                 default=1e-4,
                                 type=float,
                                 help='学习率')
        self.parser.add_argument('--num_epochs',
                                 default=24,
                                 type=int,
                                 help='训练轮数')
        self.parser.add_argument('--scheduler_step_size',
                                 default=20,
                                 type=int,
                                 help='优化轮数（从该轮之后，学习率开始变化）')

        self.parser.add_argument('--v1_multiscale',
                                 default=False,
                                 action='store_true',
                                 help='(可选项)若设置就使用monodepth1的多尺度')
        self.parser.add_argument('--avg_reprojection',
                                 default=False,
                                 action='store_true',
                                 help='(可选项)若设置就使用monodepth1的平均重投影误差')
        self.parser.add_argument('--disable_automasking',
                                 default=False,
                                 action="store_true",
                                 help='(可选项)若设置就不使用automasking')
        self.parser.add_argument('--predictive_mask',
                                 default=False,
                                 action="store_true",
                                 help='(可选项)若设置就使用predictive_mask')
        self.parser.add_argument('--no_ssim',
                                 default=False,
                                 action="store_true",
                                 help='(可选项)若设置就不使用SSIM在计算loss时候')
        self.parser.add_argument('--weights_init',
                                 default='pretrained',
                                 choices=['pretrained', 'scratch'],
                                 type=str,
                                 help='(可选项)若设置就使用pretrained or scratch')
        self.parser.add_argument('--pose_model_input',
                                 default='pairs',
                                 choices=['pairs', 'all'],
                                 type=str,
                                 help='pose网络一次输入多少图像')
        self.parser.add_argument('--pose_model_type',
                                 default='separate_resnet',
                                 choices=['posecnn', 'separate_resnet', 'shared'],
                                 type=str,
                                 help='pose网络是否共享权重')


        self.parser.add_argument('--no_cuda',
                                 default=False,
                                 action="store_true",
                                 help='(可选项)若设置就不使用CUDA')
        self.parser.add_argument('--num_workers',
                                 default=12,
                                 type=int,
                                 help='数据加载使用的线程数目')


        self.parser.add_argument('--load_weights_folder',
                                 type=str,
                                 help='加载模型的文件夹')
        self.parser.add_argument('--models_to_load',
                                 default=['DepthEncoder', 'DepthDecoder', 'PoseEncoder', 'PoseDecoder','FeatureEncoder','FeatureDecoder'],
                                 nargs='+',
                                 type=str,
                                 help='加载的模型组件')


        self.parser.add_argument('--log_frequency',
                                 default=100,
                                 type=int,
                                 help='记录日志的频率')
        self.parser.add_argument('--save_frequency',
                                 default=2,
                                 type=int,
                                 help='保存模型的频率')


        self.parser.add_argument('--eval_stereo',
                                 default=False,
                                 action="store_true",
                                 help='(可选项)若设置就在双目模式评估')
        self.parser.add_argument('--eval_mono',
                                 default=False,
                                 action="store_true",
                                 help='(可选项)若设置就在单目模式评估')
        self.parser.add_argument('--disable_median_scaling',
                                 default=False,
                                 action="store_true",
                                 help='(可选项)若设置就在评估阶段不使用中值尺度')
        self.parser.add_argument('--pred_depth_scale_factor',
                                 default=1.0,
                                 type=float,
                                 help='评估时候将估计的深度乘以这个值')
        self.parser.add_argument('--ext_disp_to_eval',
                                 type=str,
                                 help='使用.npy文件用于评估')
        self.parser.add_argument('--eval_split',
                                 default='eigen',
                                 choices=['cityscapes', 'eigen', 'eigen_benchmark', 'benchmark', 'odom_9', 'odom_10'],
                                 type=str,
                                 help='在评估时候使用的数据集分割方法')
        self.parser.add_argument('--save_pred_disps',
                                 default=False,
                                 action="store_true",
                                 help='(可选项)若设置就在评估时存储预测的视差图')
        self.parser.add_argument('--no_eval',
                                 default=False,
                                 action="store_true",
                                 help='(可选项)若设置就不使用评估')
        self.parser.add_argument('--eval_eigen_to_benchmark',
                                 action="store_true",
                                 help='(可选项)if set assume we are loading eigen results from npy but we want to evaluate using the new benchmark.')
        self.parser.add_argument('--eval_out_dir',
                                 action="store_true",
                                 help='(可选项)若设置就将评估结果输出到这个文件夹')
        self.parser.add_argument('--post_process',
                                 action="store_true",
                                 help='(可选项)if set will perform the flipping post processing from the original monodepth paper.')


    def parse(self):
        self.options = self.parser.parse_args()
        print(self.options)
        return self.options
