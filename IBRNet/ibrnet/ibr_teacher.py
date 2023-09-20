from config import config_parser
from ibrnet.model import IBRNetModel
from ibrnet.projection import Projector
from ibrnet.sample_ray import RaySamplerSingleImage
from ibrnet.render_image import render_single_image


class IBRTeacher(object):
    def __init__(self, args, src_rgbs):
        self.network = IBRNetModel(args, load_scheduler=False, load_opt=False)
        self.network.switch_to_eval()
        self.projector = Projector(device="cuda:0")
        self.featuremaps = self.network.feature_net(src_rgbs)
        self.args = args

    def eval(self, data):
        """_summary_

        Args:
            data:
            "rgb": [H,W,3]
            "camera": [34,], [H,W,intrinsic.flatten(), extrinsic.flatten()]
            "rgb_path": rgb_path
            "src_cameras": [N, 34]
            "depth_range": [2,]

        Returns:
            _type_: _description_
        """
        ray_sampler = RaySamplerSingleImage(data, device="cuda:0")
        ray_batch = ray_sampler.get_all()

        ret = render_single_image(
            ray_sampler=ray_sampler,
            ray_batch=ray_batch,
            model=self.network,
            projector=self.projector,
            chunk_size=self.args.chunk_size,
            det=True,
            N_samples=self.args.N_samples,
            inv_uniform=self.args.inv_uniform,
            N_importance=self.args.N_importance,
            white_bkgd=self.args.white_bkgd,
            featmaps=self.featmaps,
        )

        return ret
