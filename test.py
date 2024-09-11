class CameraInfo:
    def __init__(self, image_name):
        self.image_name = image_name

cam_infos_unsorted = [
    CameraInfo("1_left.jpg"),
    CameraInfo("2_right.jpg"),
    CameraInfo("1_right.jpg"),
    CameraInfo("2_left.jpg"),
]

cam_infos = sorted(cam_infos_unsorted.copy(), key=lambda x: x.image_name)


for cam_info in cam_infos:
    print(cam_info.image_name)

class CameraInfo(NamedTuple):
    uid: int            # index, Intrinsics
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    image: np.array
    image_path: str
    image_name: str
    width: int
    height: int
    fid: float          # frame time
    depth: Optional[np.array] = None


class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    train_cameras: list
    test_cameras: list
    nerf_normalization: dict
    ply_path: str

print('hello world!')