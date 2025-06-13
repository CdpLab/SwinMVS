import argparse
import cv2
import gc
import os
import sys
import time

import torch.nn.parallel
from PIL import Image
from plyfile import PlyData, PlyElement
from torch.utils.data import Dataset, DataLoader, SequentialSampler

import datasets.data_loaders as module_data
import misc.fusion as fusion
from base.parse_config import ConfigParser
from datasets.data_io import read_pfm, save_pfm
from misc.gipuma import gipuma_filter
from utils import *

# cudnn.benchmark = True

parser = argparse.ArgumentParser(description='Predict depth, filter, and fuse')
parser.add_argument('--model', default='mvsnet', help='select model')
parser.add_argument('--device', default=None, type=str, help='indices of GPUs to enable (default: all)')
parser.add_argument('--config', default=None, type=str, help='config file path (default: None)')

parser.add_argument('--dataset', default='dtu', help='select dataset')
parser.add_argument('--testpath', help='testing data dir for some scenes')
parser.add_argument('--testpath_single_scene', help='testing data path for single scene')
parser.add_argument('--testlist', help='testing scene list')
parser.add_argument('--exp_name', type=str, default=None)

parser.add_argument('--batch_size', type=int, default=1, help='testing batch size')
parser.add_argument('--numdepth', type=int, default=192, help='the number of depth values')

parser.add_argument('--resume', default=None, help='load a specific checkpoint')
parser.add_argument('--outdir', default='/SATA2/JIAQI/ViTMVSNet/output', help='output dir')
parser.add_argument('--display', action='store_true', help='display depth images and masks')

parser.add_argument('--share_cr', action='store_true', help='whether share the cost volume regularization')

parser.add_argument('--ndepths', type=str, default=None, help='ndepths')
parser.add_argument('--depth_interals_ratio', type=str, default=None, help='depth_interals_ratio')
parser.add_argument('--cr_base_chs', type=str, default="8,8,8", help='cost regularization base channels')
parser.add_argument('--grad_method', type=str, default="detach", choices=["detach", "undetach"], help='grad method')
parser.add_argument('--no_refinement', action="store_true", help='depth refinement in last stage')
parser.add_argument('--full_res', action="store_true", help='full resolution prediction')

parser.add_argument('--interval_scale', type=float, required=True, help='the depth interval scale')
parser.add_argument('--num_view', type=int, default=5, help='num of view')
parser.add_argument('--max_h', type=int, default=864, help='testing max h')
parser.add_argument('--max_w', type=int, default=1152, help='testing max w')
parser.add_argument('--fix_res', action='store_true', help='scene all using same res')
parser.add_argument('--depth_scale', type=float, default=1.0, help='depth scale')
parser.add_argument('--temperature', type=float, default=0.01, help='temperature of softmax')

parser.add_argument('--num_worker', type=int, default=4, help='depth_filer worker')
parser.add_argument('--save_freq', type=int, default=20, help='save freq of local pcd')

parser.add_argument('--filter_method', type=str, default='gipuma', choices=["gipuma", "pcd", "dpcd"],
                    help="filter method")


# filter
parser.add_argument('--prob_threshold', type=str, default='0.5,0.5,0.5,0.5', help='prob confidence')
parser.add_argument('--thres_view', type=int, default=3, help='threshold of num view')
parser.add_argument('--thres_disp', type=float, default=1.0, help='threshold of disparity')
parser.add_argument('--downsample', type=float, default=None, help='downsampling point cloud')

## dpcd filter
parser.add_argument('--dist_base', type=float, default=4.0, help='threshold of disparity')
parser.add_argument('--rel_diff_base', type=float, default=1300.0, help='downsampling point cloud')

# filter by gimupa
parser.add_argument('--fusibile_exe_path', type=str, default='./fusibile/fusibile')
parser.add_argument('--disp_threshold', type=float, default='0.2')
parser.add_argument('--num_consistent', type=float, default='3')

# tank templet
parser.add_argument('--use_short_range', action='store_true')

# confidence
parser.add_argument('--combine_conf', action='store_true')
parser.add_argument('--tmp', default=1.0, type=float)
parser.add_argument('--tmps', default=None, type=str)
parser.add_argument('--save_all_confs', action='store_true')

# parse arguments and check
args = parser.parse_args()
print("argv:", sys.argv[1:])
print_args(args)
if args.testpath_single_scene:
    args.testpath = os.path.dirname(args.testpath_single_scene)

Interval_Scale = args.interval_scale
print("***********Interval_Scale**********\n", Interval_Scale)

# args.outdir = args.outdir + f'_{args.max_w}x{args.max_h}'
os.makedirs(args.outdir, exist_ok=True)


# read intrinsics and extrinsics
def read_camera_parameters(filename):
    with open(filename) as f:
        lines = f.readlines()
        lines = [line.rstrip() for line in lines]
    # extrinsics: line [1,5), 4x4 matrix
    extrinsics = np.fromstring(' '.join(lines[1:5]), dtype=np.float32, sep=' ').reshape((4, 4))
    # intrinsics: line [7-10), 3x3 matrix
    intrinsics = np.fromstring(' '.join(lines[7:10]), dtype=np.float32, sep=' ').reshape((3, 3))
    # TODO: assume the feature is 1/4 of the original image size
    # intrinsics[:2, :] /= 4
    return intrinsics, extrinsics


# read an image
def read_img(filename):
    img = Image.open(filename)
    # scale 0~255 to 0~1
    np_img = np.array(img, dtype=np.float32) / 255.
    return np_img


# read a binary mask
def read_mask(filename):
    return read_img(filename) > 0.5


# save a binary mask
def save_mask(filename, mask):
    assert mask.dtype == np.bool
    mask = mask.astype(np.uint8) * 255
    Image.fromarray(mask).save(filename)


# read a pair file, [(ref_view1, [src_view1-1, ...]), (ref_view2, [src_view2-1, ...]), ...]
def read_pair_file(filename):
    data = []
    with open(filename) as f:
        num_viewpoint = int(f.readline())
        # 49 viewpoints
        for view_idx in range(num_viewpoint):
            ref_view = int(f.readline().rstrip())
            src_views = [int(x) for x in f.readline().rstrip().split()[1::2]]
            if len(src_views) > 0:
                data.append((ref_view, src_views))
    return data


def write_cam(file, cam):
    f = open(file, "w")
    f.write('extrinsic\n')
    for i in range(0, 4):
        for j in range(0, 4):
            f.write(str(cam[0][i][j]) + ' ')
        f.write('\n')
    f.write('\n')

    f.write('intrinsic\n')
    for i in range(0, 3):
        for j in range(0, 3):
            f.write(str(cam[1][i][j]) + ' ')
        f.write('\n')

    f.write(
        '\n' + str(cam[1][3][0]) + ' ' + str(cam[1][3][1]) + ' ' + str(cam[1][3][2]) + ' ' + str(cam[1][3][3]) + '\n')

    f.close()


# run model to save depth maps and confidence maps
def save_depth(testlist, config):
    # dataset, dataloader

    init_kwags = {
        "data_path": args.testpath,
        "data_list": testlist,
        "mode": "test",
        "num_srcs": args.num_view,
        "num_depths": args.numdepth,
        "interval_scale": Interval_Scale,
        "shuffle": False,
        "batch_size": 1,
        "fix_res": args.fix_res,
        "max_h": args.max_h,
        "max_w": args.max_w,
        "dataset_eval": args.dataset,
        "iterative": False,  # iterative inference
        "refine": not args.no_refinement,
        "use_short_range": args.use_short_range,
        "num_workers": 4,
    }
    test_data_loader = module_data.DTULoader(**init_kwags)

    # model
    # build models architecture, then print to console
    if config['arch']['args']['vit_args'].get('twin', False):
        from models.vitmvs_model import TwinMVSNet
        model = TwinMVSNet(config['arch']['args'])
    else:
        from models.vitmvs_model import DINOMVSNet
        model = DINOMVSNet(config['arch']['args'])

    print('Loading checkpoint: {} ...'.format(config.resume))
    checkpoint = torch.load(str(config.resume))
    state_dict = checkpoint['state_dict']
    new_state_dict = {}
    for key, val in state_dict.items():
        new_state_dict[key.replace('module.', '')] = val
    model.load_state_dict(new_state_dict, strict=True)

    # prepare models for testing
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    # temp setting
    if hasattr(model, 'vit_args') and 'height' in model.vit_args and 'width' in model.vit_args:
        model.vit_args['height'] = args.max_h // 2
        model.vit_args['width'] = args.max_w // 2

    times = []

    # get tmp
    if args.tmps is not None:
        tmp = [float(a) for a in args.tmps.split(',')]
    else:
        tmp = args.tmp

    valid_metrics = DictAverageMeter()
    valid_metrics.reset()
    with torch.no_grad():
        for batch_idx, sample in enumerate(test_data_loader):
            torch.cuda.synchronize()
            start_time = time.time()
            sample_cuda = tocuda(sample)
            num_stage = 3 if args.no_refinement else 4
            imgs, cam_params = sample_cuda["imgs"], sample_cuda["proj_matrices"]
            if args.dataset == 'dtu':
                depth_gt = sample_cuda["depth"]["stage{}".format(num_stage)]
                mask = sample_cuda["mask"]["stage{}".format(num_stage)]
            B, V, _, H, W = imgs.shape
            depth_interval = sample_cuda['depth_values'][:, 1] - sample_cuda['depth_values'][:, 0]
            filenames = sample["filename"]
            # with torch.cuda.amp.autocast():
            outputs = model.forward(imgs, cam_params, sample_cuda['depth_values'], tmp=tmp)
            torch.cuda.synchronize()

            end_time = time.time()
            times.append(end_time - start_time)
            depth_est_cuda = outputs['refined_depth']
            outputs = tensor2numpy(outputs)
            del sample_cuda

            cams = sample["proj_matrices"]["stage{}".format(num_stage)].numpy()
            # imgs = sample["imgs"].numpy()
            print('Iter {}/{}, Time:{} Res:{}'.format(batch_idx, len(test_data_loader), end_time - start_time,
                                                      outputs["refined_depth"][0].shape))

            # save depth maps and confidence maps
            for filename, cam, img, depth_est, conf_stage1, conf_stage2, conf_stage3, conf_stage4, conf_stage4_ in zip(filenames, cams, imgs, outputs["refined_depth"],
                                                                                                                       outputs["stage1"]["photometric_confidence"],
                                                                                                                       outputs["stage2"]["photometric_confidence"],
                                                                                                                       outputs["stage3"]["photometric_confidence"],
                                                                                                                       outputs["photometric_confidence"],
                                                                                                                       outputs["stage4"]["photometric_confidence"]):
                img = img[0]  # ref view
                cam = cam[0]  # ref cam
                depth_filename = os.path.join(args.outdir, filename.format('depth_est', '.pfm'))
                confidence_filename = os.path.join(args.outdir, filename.format('confidence', '.npy'))
                cam_filename = os.path.join(args.outdir, filename.format('cams', '_cam.txt'))
                img_filename = os.path.join(args.outdir, filename.format('images', '.jpg'))
                os.makedirs(depth_filename.rsplit('/', 1)[0], exist_ok=True)
                os.makedirs(confidence_filename.rsplit('/', 1)[0], exist_ok=True)
                os.makedirs(cam_filename.rsplit('/', 1)[0], exist_ok=True)
                os.makedirs(img_filename.rsplit('/', 1)[0], exist_ok=True)
                # save depth maps
                save_pfm(depth_filename, depth_est)
                h, w = depth_est.shape[0], depth_est.shape[1]
                # save confidence maps
                if args.combine_conf:
                    photometric_confidence = conf_stage4
                    if args.save_all_confs:  # only for visualization
                        all_confidence_filename = os.path.join(args.outdir, filename.format('confidence_all', '.npy'))
                        os.makedirs(all_confidence_filename.rsplit('/', 1)[0], exist_ok=True)
                        all_photometric_confidence = np.stack(
                            [conf_stage1, conf_stage2, conf_stage3, conf_stage4_]).transpose([1, 2, 0])
                        np.save(all_confidence_filename, all_photometric_confidence)
                else:
                    conf_stage1 = cv2.resize(conf_stage1, (w, h), interpolation=cv2.INTER_NEAREST)
                    conf_stage2 = cv2.resize(conf_stage2, (w, h), interpolation=cv2.INTER_NEAREST)
                    conf_stage3 = cv2.resize(conf_stage3, (w, h), interpolation=cv2.INTER_NEAREST)
                    photometric_confidence = np.stack([conf_stage1, conf_stage2, conf_stage3, conf_stage4_]).transpose([1, 2, 0])
                np.save(confidence_filename, photometric_confidence)

                std = torch.tensor([0.229, 0.224, 0.225], device=img.device).reshape((3, 1, 1))
                mean = torch.tensor([0.485, 0.456, 0.406], device=img.device).reshape((3, 1, 1))
                img = img * std + mean
                img = img.permute(1, 2, 0).cpu().numpy()

                write_cam(cam_filename, cam)
                img = np.clip(img * 255, 0, 255).astype(np.uint8)

                img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                cv2.imwrite(img_filename, img_bgr)

                if args.dataset == 'dtu':
                    di = depth_interval[0].item() / 2.65
                    scalar_outputs = {"abs_depth_error": AbsDepthError_metrics(depth_est_cuda, depth_gt, mask > 0.5),
                                      "thres1mm_error": Thres_metrics(depth_est_cuda, depth_gt, mask > 0.5, di),
                                      "thres2mm_error": Thres_metrics(depth_est_cuda, depth_gt, mask > 0.5, di * 2),
                                      "thres4mm_error": Thres_metrics(depth_est_cuda, depth_gt, mask > 0.5, di * 4),
                                      "thres8mm_error": Thres_metrics(depth_est_cuda, depth_gt, mask > 0.5, di * 8),
                                      "thres14mm_error": Thres_metrics(depth_est_cuda, depth_gt, mask > 0.5, di * 14),
                                      "thres20mm_error": Thres_metrics(depth_est_cuda, depth_gt, mask > 0.5, di * 20)}
                    scalar_outputs = tensor2float(scalar_outputs)
                    valid_metrics.update(scalar_outputs)

    print("average time: ", sum(times) / len(times))
    if args.dataset == 'dtu':
        valid_metrics = valid_metrics.mean()
        with open(os.path.join(args.outdir, 'depth_metric.txt'), 'w') as w:
            for k in valid_metrics:
                w.write(k + ' ' + str(valid_metrics[k]) + '\n')
    torch.cuda.empty_cache()
    gc.collect()


class TTDataset(Dataset):
    def __init__(self, pair_folder, scan_folder, n_src_views=10):
        super(TTDataset, self).__init__()
        pair_file = os.path.join(pair_folder, "pair.txt")
        self.scan_folder = scan_folder
        self.pair_data = read_pair_file(pair_file)
        self.n_src_views = n_src_views

    def __len__(self):
        return len(self.pair_data)

    def __getitem__(self, idx):
        id_ref, id_srcs = self.pair_data[idx]
        id_srcs = id_srcs[:self.n_src_views]

        ref_intrinsics, ref_extrinsics = read_camera_parameters(
            os.path.join(self.scan_folder, 'cams/{:0>8}_cam.txt'.format(id_ref)))
        ref_cam = np.zeros((2, 4, 4), dtype=np.float32)
        ref_cam[0] = ref_extrinsics
        ref_cam[1, :3, :3] = ref_intrinsics
        ref_cam[1, 3, 3] = 1.0
        # load the reference image
        ref_img = read_img(os.path.join(self.scan_folder, 'images/{:0>8}.jpg'.format(id_ref)))
        ref_img = ref_img.transpose([2, 0, 1])
        # load the estimated depth of the reference view
        ref_depth_est = read_pfm(os.path.join(self.scan_folder, 'depth_est/{:0>8}.pfm'.format(id_ref)))[0]
        ref_depth_est = np.array(ref_depth_est, dtype=np.float32)
        # load the photometric mask of the reference view
        # confidence = read_pfm(os.path.join(self.scan_folder, 'confidence/{:0>8}.pfm'.format(id_ref)))[0]
        conf_path = os.path.join(self.scan_folder, 'confidence/{:0>8}.npy'.format(id_ref))
        if not os.path.exists(conf_path):
            conf_path = os.path.join(self.scan_folder, 'confidence_v2/{:0>8}.npy'.format(id_ref))
        confidence = np.load(conf_path)
        if not args.combine_conf:
            confidence = np.array(confidence, dtype=np.float32).transpose([2, 0, 1])

        src_depths, src_confs, src_cams = [], [], []
        for ids in id_srcs:
            if not os.path.exists(os.path.join(self.scan_folder, 'cams/{:0>8}_cam.txt'.format(ids))):
                continue
            src_intrinsics, src_extrinsics = read_camera_parameters(
                os.path.join(self.scan_folder, 'cams/{:0>8}_cam.txt'.format(ids)))
            src_proj = np.zeros((2, 4, 4), dtype=np.float32)
            src_proj[0] = src_extrinsics
            src_proj[1, :3, :3] = src_intrinsics
            src_proj[1, 3, 3] = 1.0
            src_cams.append(src_proj)
            # the estimated depth of the source view
            src_depth_est = read_pfm(os.path.join(self.scan_folder, 'depth_est/{:0>8}.pfm'.format(ids)))[0]
            src_depths.append(np.array(src_depth_est, dtype=np.float32))
            # src_conf = read_pfm(os.path.join(self.scan_folder, 'confidence/{:0>8}.pfm'.format(ids)))[0]
            conf_path = os.path.join(self.scan_folder, 'confidence/{:0>8}.npy'.format(ids))
            if not os.path.exists(conf_path):
                conf_path = os.path.join(self.scan_folder, 'confidence_v2/{:0>8}.npy'.format(ids))
            src_conf = np.load(conf_path)
            if not args.combine_conf:
                src_confs.append(np.array(src_conf, dtype=np.float32).transpose([2, 0, 1]))
            else:
                src_confs.append(src_conf)
        src_depths = np.expand_dims(np.stack(src_depths, axis=0), axis=1)
        src_confs = np.stack(src_confs, axis=0)
        src_cams = np.stack(src_cams, axis=0)
        return {"ref_depth": np.expand_dims(ref_depth_est, axis=0),
                "ref_cam": ref_cam,
                "ref_conf": confidence,  # np.expand_dims(confidence, axis=0),
                "src_depths": src_depths,
                "src_cams": src_cams,
                "src_confs": src_confs,
                "ref_img": ref_img,
                "ref_id": id_ref}


def filter_depth(pair_folder, scan_folder, out_folder, plyfilename):
    tt_dataset = TTDataset(pair_folder, scan_folder, n_src_views=10)
    sampler = SequentialSampler(tt_dataset)
    tt_dataloader = DataLoader(tt_dataset, batch_size=1, shuffle=False, sampler=sampler, num_workers=2,
                               pin_memory=True, drop_last=False)
    views = {}
    prob_threshold = args.prob_threshold
    prob_threshold = [float(p) for p in prob_threshold.split(',')]
    for batch_idx, sample_np in enumerate(tt_dataloader):
        sample = tocuda(sample_np)
        for ids in range(sample["src_depths"].size(1)):
            if args.combine_conf:
                src_prob_mask = sample['src_confs'][:, ids] > prob_threshold[0]
            else:
                src_prob_mask = fusion.prob_filter(sample['src_confs'][:, ids, ...], prob_threshold)
            sample["src_depths"][:, ids, ...] *= src_prob_mask.float()

        if args.combine_conf:
            prob_mask = sample['ref_conf'] > prob_threshold[0]
        else:
            prob_mask = fusion.prob_filter(sample['ref_conf'], prob_threshold)

        reproj_xyd, in_range = fusion.get_reproj(
            *[sample[attr] for attr in ['ref_depth', 'src_depths', 'ref_cam', 'src_cams']])
        vis_masks, vis_mask = fusion.vis_filter(sample['ref_depth'], reproj_xyd, in_range, args.thres_disp, 0.01,
                                                args.thres_view)

        ref_depth_ave = fusion.ave_fusion(sample['ref_depth'], reproj_xyd, vis_masks)

        mask = fusion.bin_op_reduce([prob_mask, vis_mask], torch.min)

        idx_img = fusion.get_pixel_grids(*ref_depth_ave.size()[-2:]).unsqueeze(0)
        idx_cam = fusion.idx_img2cam(idx_img, ref_depth_ave, sample['ref_cam'])
        points = fusion.idx_cam2world(idx_cam, sample['ref_cam'])[..., :3, 0].permute(0, 3, 1, 2)

        points_np = points.cpu().data.numpy()
        mask_np = mask.cpu().data.numpy().astype(np.bool)
        # dir_vecs = dir_vecs.cpu().data.numpy()
        ref_img = sample_np['ref_img'].data.numpy()
        for i in range(points_np.shape[0]):
            print(np.sum(np.isnan(points_np[i])))
            p_f_list = [points_np[i, k][mask_np[i, 0]] for k in range(3)]
            p_f = np.stack(p_f_list, -1)
            c_f_list = [ref_img[i, k][mask_np[i, 0]] for k in range(3)]
            c_f = np.stack(c_f_list, -1) * 255
            # d_f_list = [dir_vecs[i, k][mask_np[i, 0]] for k in range(3)]
            # d_f = np.stack(d_f_list, -1)
            ref_id = str(sample_np['ref_id'][i].item())
            views[ref_id] = (p_f, c_f.astype(np.uint8))
            print("processing {}, ref-view{:0>2}, photo/geo/final-mask:{}/{}/{}".format(scan_folder, int(ref_id),
                                                                                        prob_mask[i].float().mean().item(),
                                                                                        vis_mask[i].float().mean().item(),
                                                                                        mask[i].float().mean().item()))

    print('Write combined PCD')
    p_all, c_all = [np.concatenate([v[k] for key, v in views.items()], axis=0) for k in range(2)]

    vertexs = np.array([tuple(v) for v in p_all], dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
    vertex_colors = np.array([tuple(v) for v in c_all], dtype=[('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])

    vertex_all = np.empty(len(vertexs), vertexs.dtype.descr + vertex_colors.dtype.descr)
    for prop in vertexs.dtype.names:
        vertex_all[prop] = vertexs[prop]
    for prop in vertex_colors.dtype.names:
        vertex_all[prop] = vertex_colors[prop]

    el = PlyElement.describe(vertex_all, 'vertex')
    PlyData([el]).write(plyfilename)
    print("saving the final model to", plyfilename)


def dynamic_filter_depth(pair_folder, scan_folder, out_folder, plyfilename):
    tt_dataset = TTDataset(pair_folder, scan_folder, n_src_views=10)
    sampler = SequentialSampler(tt_dataset)
    tt_dataloader = DataLoader(tt_dataset, batch_size=1, shuffle=False, sampler=sampler, num_workers=2,
                               pin_memory=True, drop_last=False)
    views = {}
    prob_threshold = args.prob_threshold
    prob_threshold = [float(p) for p in prob_threshold.split(',')]
    for batch_idx, sample_np in enumerate(tt_dataloader):
        num_src_views = sample_np['src_depths'].shape[1]
        dy_range = num_src_views + 1  # 10
        sample = tocuda(sample_np)

        if args.combine_conf:
            prob_mask = sample['ref_conf'] > prob_threshold[0]
        else:
            prob_mask = fusion.prob_filter(sample['ref_conf'], prob_threshold)

        ref_depth = sample['ref_depth']  # [n 1 h w ]
        device = ref_depth.device
        reproj_xyd = fusion.get_reproj_dynamic(
            *[sample[attr] for attr in ['ref_depth', 'src_depths', 'ref_cam', 'src_cams']])
        # reproj_xyd   nv 3 h w

        # 4 1300
        vis_masks, vis_mask = fusion.vis_filter_dynamic(sample['ref_depth'], reproj_xyd, dist_base=args.dist_base, rel_diff_base=args.rel_diff_base)

        # mask reproj_depth
        reproj_depth = reproj_xyd[:, :, -1]  # [1 v h w]
        reproj_depth[~vis_mask.squeeze(2)] = 0  # [n v h w ]
        geo_mask_sums = vis_masks.sum(dim=1)  # 0~v
        geo_mask_sum = vis_mask.sum(dim=1)
        depth_est_averaged = (torch.sum(reproj_depth, dim=1, keepdim=True) + ref_depth) / (geo_mask_sum + 1)  # [1,1,h,w]
        geo_mask = geo_mask_sum >= dy_range  # all zero
        for i in range(2, dy_range):
            geo_mask = torch.logical_or(geo_mask, geo_mask_sums[:, i - 2] >= i)

        mask = fusion.bin_op_reduce([prob_mask, geo_mask], torch.min)
        idx_img = fusion.get_pixel_grids(*depth_est_averaged.size()[-2:]).unsqueeze(0)
        idx_cam = fusion.idx_img2cam(idx_img, depth_est_averaged, sample['ref_cam'])
        points = fusion.idx_cam2world(idx_cam, sample['ref_cam'])[..., :3, 0].permute(0, 3, 1, 2)

        points_np = points.cpu().data.numpy()
        mask_np = mask.cpu().data.numpy().astype(np.bool)

        ref_img = sample_np['ref_img'].data.numpy()
        for i in range(points_np.shape[0]):
            print(np.sum(np.isnan(points_np[i])))
            p_f_list = [points_np[i, k][mask_np[i, 0]] for k in range(3)]
            p_f = np.stack(p_f_list, -1)
            c_f_list = [ref_img[i, k][mask_np[i, 0]] for k in range(3)]
            c_f = np.stack(c_f_list, -1) * 255

            ref_id = str(sample_np['ref_id'][i].item())
            views[ref_id] = (p_f, c_f.astype(np.uint8))
            print("processing {}, ref-view{:0>2}, photo/geo/final-mask:{}/{}/{}".format(scan_folder, int(ref_id),
                                                                                        prob_mask[i].float().mean().item(),
                                                                                        geo_mask[i].float().mean().item(),
                                                                                        mask[i].float().mean().item()))

    print('Write combined PCD')
    p_all, c_all = [np.concatenate([v[k] for key, v in views.items()], axis=0) for k in range(2)]

    vertexs = np.array([tuple(v) for v in p_all], dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
    vertex_colors = np.array([tuple(v) for v in c_all], dtype=[('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])

    vertex_all = np.empty(len(vertexs), vertexs.dtype.descr + vertex_colors.dtype.descr)
    for prop in vertexs.dtype.names:
        vertex_all[prop] = vertexs[prop]
    for prop in vertex_colors.dtype.names:
        vertex_all[prop] = vertex_colors[prop]

    el = PlyElement.describe(vertex_all, 'vertex')
    PlyData([el]).write(plyfilename)
    print("saving the final model to", plyfilename)


def pcd_filter_worker(scan):
    save_name = '{}.ply'.format(scan)
    pair_folder = os.path.join(args.testpath, scan)
    scan_folder = os.path.join(args.outdir, scan)
    out_folder = os.path.join(args.outdir, scan)
    if args.filter_method == 'pcd':
        filter_depth(pair_folder, scan_folder, out_folder, os.path.join(args.outdir, save_name))
    else:
        dynamic_filter_depth(pair_folder, scan_folder, out_folder, os.path.join(args.outdir, save_name))


def pcd_filter(testlist):
    for scan in testlist:
        pcd_filter_worker(scan)


if __name__ == '__main__':
    config = ConfigParser.from_args(parser, mkdir=False)
    if args.ndepths is not None:
        config['arch']['args']['ndepths'] = [int(d) for d in args.ndepths.split(',')]
    if args.depth_interals_ratio is not None:
        config['arch']['args']['depth_interals_ratio'] = [float(d) for d in args.depth_interals_ratio.split(',')]

    if args.testlist != "all":
        with open(args.testlist) as f:
            content = f.readlines()
            testlist = [line.rstrip() for line in content]
    else:
        # for tanks & temples or eth3d or colmap
        testlist = [e for e in os.listdir(args.testpath) if os.path.isdir(os.path.join(args.testpath, e))] \
            if not args.testpath_single_scene else [os.path.basename(args.testpath_single_scene)]

    # step1. save all the depth maps and the masks in outputs directory
    save_depth(testlist, config)

    # step2. filter saved depth maps with photometric confidence maps and geometric constraints
    if args.filter_method == "pcd" or args.filter_method == "dpcd":
        # support multi-processing, the default number of worker is 4
        pcd_filter(testlist)

    elif args.filter_method == 'gipuma':
        prob_threshold = args.prob_threshold
        prob_threshold = [float(p) for p in prob_threshold.split(',')]
        gipuma_filter(testlist, args.outdir, prob_threshold, args.disp_threshold, args.num_consistent,
                      args.fusibile_exe_path)
    else:
        raise NotImplementedError
