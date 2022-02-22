import csv
from plyfile import PlyData, PlyElement
import numpy as np
import os
import trimesh
import glob
import pandas as pd
import json


def get_label_bbox(label,grid_shape):
    nonzero_label = np.argwhere((label.reshape(grid_shape) != 0))
    xmax,ymax,zmax=nonzero_label.max(0)
    xmin,ymin,zmin=nonzero_label.min(0)
    cornerpts = np.array([ [xmin,ymin,  zmin],
                  [xmin,ymin, zmax],
                  [xmin,ymax,  zmax],
                  [xmin,ymax,zmin],
                [xmax,ymax,  zmax],
                  [xmax,ymin,  zmin],
                  [xmax,ymax,  zmin],
                  [xmax,ymin,zmax]])
    grid_inds = (np.indices(grid_shape).reshape(3, -1).T)

    cond = get_points_inside_boxes(grid_inds,[cornerpts])
    label[cond == False] = 255 
    return label

def get_points_inside_boxes(points, boxes):
    """[summary]

    Args:
        points ([np.array]): [N x 3 point cloud]
        boxes ([type]): [list of 8x3 boxes ]

    Returns:
        [np.array]: [boolean mask for gettig the points inside given boxes\]
    """
    cond = np.zeros(len(points))
    for box in boxes:
        
        maxx, maxy, maxz = box.max(0)

        minx, miny, minz = box.min(0)

        xs = points[:, 0]
        ys = points[:, 1]
        zs = points[:, 2]
        cond = np.logical_or(cond, (xs <= maxx) & (xs >= minx) & (
            ys <= maxy) & (ys >= miny) & (zs <= maxz) & (zs >= minz))
    return cond


def get_inside_grid(pts, grid_shape):
    """[summary]

    Args:
        pts ([type]): [Point cloud in grid coordinates]
        grid_shape ([type]): [Dimensions of the voxel grid]
    Returns:
        [np.array] : [boolean mask for filtering points that fall outside the grid]
    """
    xmask = (pts[:, 0] < grid_shape[0]) & (pts[:, 0] > 0)
    ymask = (pts[:, 1] < grid_shape[1]) & (pts[:, 1] > 0)
    zmask = (pts[:, 2] < grid_shape[2]) & (pts[:, 2] > 0)
    inside_grid_mask = xmask & ymask & zmask
    return inside_grid_mask


def export_grid(outname,resh,ignore=[255,0]):
    """[Exports given grid to a point cloud]

    Args:
        outname ([str]): [Output file name]
        resh ([np.array]): [3D Volume in grid representation]
        ignore (list, optional): [values to ignore]. Defaults to [255,0].
    Returns:
        Grid as a point cloud and its colors
    """
    pts = []
    colors = []
    grid_shape = resh.shape
    # print(grid_shape)
    colorMap = np.array([[255, 0, 0],
                                  [0, 255, 0],
                                  [0, 0, 255],
                                  [80, 128, 255],
                                  [255, 230, 180],
                                  [255, 0, 255],
                                  [0, 255, 255],
                                  [100, 0, 0],
                                  [0, 100, 0],
                                  [255, 255, 0],
                                  [50, 150, 0],
                                  [200, 255, 255],
                                  [255, 200, 255],
                                  [128, 128, 80],
                                  [0, 50, 128],
                                  [0, 100, 100],
                                  [0, 255, 128],
                                  [0, 128, 255],
                                  [255, 0, 128],
                                  [128, 0, 255],
                                  [255, 128, 0],
                                  [128, 255, 0],
        ])
#     resh = np.load("results_overfit_scannet_corrected/0000.npy").reshape(grid_shape)
    for x in range(grid_shape[0]):
        for y in range(grid_shape[1]):
            for z in range(grid_shape[2]):
                if resh[x,y,z] not in ignore:
                    colors.append(colorMap[resh[x,y,z]])
                    pts.append([x,y,z])


    
    a = trimesh.points.PointCloud(pts,colors).export(outname)

    return np.array(pts),np.array(colors)
def export_tsdf(outname,resh,ignore=[255,0]):
    """[summary]
    Same as export grid but for tsdf and only negative values. Does not return anything
    Args:
        outname ([type]): [description]
        resh ([type]): [description]
        ignore (list, optional): [description]. Defaults to [255,0].
    """
    pts = []
    grid_shape = resh.shape
#     resh = np.load("results_overfit_scannet_corrected/0000.npy").reshape(grid_shape)
    for x in range(grid_shape[0]):
        for y in range(grid_shape[1]):
            for z in range(grid_shape[2]):
                if resh[x,y,z] != 0:
                #                     colors.append(colorMap[resh[x,y,z]])
                    pts.append([x,y,z])


    
    a = trimesh.points.PointCloud(pts).export(outname)  

def box_filter_label_mapping(boxes,VOX_SIZE,ptrans):
    """[Converts the boxes from world coordinates to grid coordinates]

    Args:
        boxes ([type]): [list of 8x3 boxes]
        VOX_SIZE ([type]): [description]
        ptrans ([type]): [translation vector to carry input pcd to center of the grid]

    Returns:
        [type]: [New boxes in grid coords]
    """
    new_boxes = []
    for v in boxes:
        v = (v[:, [0, 1, 2]]/VOX_SIZE)
        v+=ptrans
        v=v.astype(int)
        new_boxes.append(v)
    return new_boxes
def frame_to_grid(pts, grid_shape, VOX_SIZE):
    """[Converts world coordinates to grid coordinates]

    Args:
        pts ([type]): [Nx3 point cloud]
        grid_shape ([type]): [description]
        VOX_SIZE ([type]): [description]
    """
    pts = (pts[:, [0, 1, 2]]/VOX_SIZE)

    ptrans = grid_shape/2 - pts.mean(0)
    pts = ((pts + ptrans)).astype(int)

    return pts, ptrans


def get_mapping(pts, pts2d, grid_shape):
    """[Computes the mapping of 3d points to image coordinates]

    Args:
        pts ([type]): [3d points in grid coordinates]
        pts2d ([type]): [   2d indices of 3d points]
        grid_shape ([type]): [Shape of volume]
    """
    mapping = np.zeros(grid_shape) + 307200
    inside_grid_mask = get_inside_grid(pts, grid_shape)

    mapping[pts[inside_grid_mask][:, 0], pts[inside_grid_mask][:, 1],
            pts[inside_grid_mask][:, 2]] = pts2d[inside_grid_mask]

    return mapping.flatten()
def get_camera_pose(curr_pose):
    """[reads the current camera pose from txt file]

    Args:
        curr_pose ([type]): [pose filename]

    Returns:
        [type]: [4x4 matrix for camera pose ]
    """
    camera_pose = pd.read_csv(curr_pose, header=None,
                              delimiter=" ").values[:, :-1]
    return camera_pose


def get_grid_to_camera(camera_pose, axis_align_matrix):
    """[Transformation from world coords to camera]

    Args:
        camera_pose ([type]): [description]
        axis_align_matrix ([type]): [description]

    Returns:
        [type]: [4x4 transformation matrix]
    """
    trns = np.matmul(np.linalg.inv(camera_pose),
                     np.linalg.inv(axis_align_matrix))
    return trns


def get_axis_aligned_matrix(scene_name, scan_dir="/home/yildirir/workspace/votenet/scannet/scans/"):
    """[Reads axis align matrix from scene metadata]

    Args:
        scene_name ([type]): [description]
        scan_dir (str, optional): [description]. Defaults to "/home/yildirir/workspace/votenet/scannet/scans/".

    Returns:
        [type]: [The axis align matrix]
    """
    # scan_dir = "/home/yildirir/workspace/votenet/scannet/scans/" #get this from config

    scan_path = os.path.join(scan_dir, scene_name)
    meta_file = os.path.join(scan_path, "{}.txt".format(scene_name))

    lines = open(meta_file).readlines()

    for line in lines:
        if 'axisAlignment' in line:
            axis_align_matrix = [float(x) for x in line.rstrip().strip(
                'axisAlignment = ').split(' ')]
            break

    axis_align_matrix = np.array(axis_align_matrix).reshape((4, 4))
    return axis_align_matrix


def get_color_extrinsic(scene_name, scan_dir="/home/yildirir/workspace/votenet/scannet/scans/"):
    # scan_dir = "/home/yildirir/workspace/votenet/scannet/scans/"  # get this from config
    """[Returns the color extrinsic matrix from the metadata]

    Args:
        scene_name ([type]): [description]
        scan_dir (str, optional): [description]. Defaults to "/home/yildirir/workspace/votenet/scannet/scans/".
    """
    scan_path = os.path.join(scan_dir, scene_name)
    meta_file = os.path.join(scan_path, "{}.txt".format(scene_name))

    lines = open(meta_file).readlines()
    for line in lines:
        if 'colorToDepthExtrinsics' in line:
            colorToDepthExtr = [float(x) for x in line.rstrip().strip(
                'colorToDepthExtrinsics = ').split(' ')]
            break
    colorToDepthExtr = np.array(colorToDepthExtr).reshape((4, 4))


def get_instance_boxes(instancedir,with_classes=False, thresh=0.3):
    """[Reads the instance boxes from dataset, filter by threshold]

    Args:
        instancedir ([type]): [description]
        thresh (float, optional): [description]. Defaults to 0.3.

    Returns:
        [type]: [description]
    """
    instances = np.array(sorted(os.listdir(instancedir)))
    scores = np.array([float(a[:-4].split("_")[5]) for a in instances])
    classes =  np.array([float(a[:-4].split("_")[3]) for a in instances])
    # print(classes,scores)
    boxes = [trimesh.load(os.path.join(instancedir, i)).bounding_box.vertices for i in instances[scores > thresh]]
    # DONE: get the classses here
    if with_classes == False:
        return boxes
    else:
        return boxes, classes[scores > thresh]
def get_inside_box_mask(points, box):
    """[summary]

    Args:
        points ([np.array]): [N x 3 point cloud]
        boxes ([type]): [list of 8x3 boxes ]

    Returns:
        [np.array]: [boolean mask for gettig the points inside given boxes\]
    """
    cond = np.zeros(len(points))
        
    maxx, maxy, maxz = box.max(0)

    minx, miny, minz = box.min(0)

    xs = points[:, 0]
    ys = points[:, 1]
    zs = points[:, 2]
    
    cond = np.logical_or(cond, (xs <= maxx) & (xs >= minx) & (
        ys <= maxy) & (ys >= miny) & (zs <= maxz) & (zs >= minz))
    return cond

def corners_to_obb(boxes,classes):
    new_boxes = []
    for box,cls in zip(boxes,classes):
        x,y,z = box.mean(0)
        
        w,h,d = box.max(0) - box.min(0) # TODO: this order is not correct probably
        new_boxes.append(np.array([x,y,z,w,h,d,cls]))
        

    return np.array(new_boxes)
def get_instance_and_semantic_pcd_from_boxes(points,boxes,classes):

    instances = np.zeros(points.shape[0]) -1
    class_ids =  np.zeros(points.shape[0]) - 1
    instance_ids = np.arange(len(boxes))
    for idx,box in enumerate(boxes):
        cond = get_inside_box_mask(points,box)
        instances[cond] = instance_ids[idx]
        class_ids[cond] = classes[idx] # TODO: make sure this is correct    
    # pass 
    return instances,class_ids    
        

def get_pcd_from_depth(depth_image, depth_intrinsic):
    """[Vectorized computation of point cloud from given depth image and camera matrix]

    Args:
        depth_image ([type]): [description]
        depth_intrinsic ([type]): [description]

    Returns:
        [type]: [description]
    """
    img_inds = np.indices(depth_image.shape).reshape(2, -1).T
    depth_structure = np.zeros([307200, 3])
    depths = depth_image[img_inds[:, 0], img_inds[:, 1]]
    depth_structure[:, 1] = img_inds[:, 0]*depths
    depth_structure[:, 0] = img_inds[:, 1]*depths
    depth_structure[:, 2] = depths
    pts3d = np.matmul(np.linalg.inv(depth_intrinsic)
                      [:3, :3], depth_structure.T).T
    valid_depth_inds = (depths > 0) & (depths < 3)
    return pts3d, valid_depth_inds


def tsdf_from_depth(grid_shape, VOX_SIZE, TRUNCATION, depth_intrinsic, depth_image, worldToCam, translate, upshift=0):
    """[TSDF computation from given input depth image]

    Args:
        grid_shape ([type]): [description]
        VOX_SIZE ([type]): [description]
        TRUNCATION ([type]): [description]
        depth_intrinsic ([type]): [description]
        depth_image ([type]): [description]
        worldToCam ([type]): [description]
        translate ([type]): [description]
        upshift (int, optional): [description]. Defaults to 0.
    """

    grid_inds = (np.indices(grid_shape).reshape(3, -1).T)
    grid = np.zeros(np.product(grid_shape))
#     print(translate)
    dh, dw = depth_image.shape
    vec1 = (grid_inds)
    vec1 = vec1 + upshift

    vec1 = (vec1 - translate)

#     vec1[:,2] *= -1
#     vec1[:,1] *= -1

    # - translate  + upshift)*VOX_SIZE# + grid_shape/2 # back to world size and to cam pos #tranlation was here
    vec1 = vec1*VOX_SIZE
#     vec1 = vec1[:,[0,2,1]]
#     vec1[:,2] *= -1

    rot = worldToCam[:3, :3]
    t = worldToCam[:-1, 3]

    vec1 = np.matmul(rot, vec1.T).T + t

    t = np.linalg.inv(worldToCam)[:-1, 3]
#     t[2] = t[2] * -1

#     t = np.array([t[0],t[2],t[1]])
#     t = t/VOX_SIZE
#     t[2] = t[2] * -1
#     t[1] = t[1] * -1
#     t = t + translate
#     t = t - upshift
#     print(t)
#     trimesh.points.PointCloud(vec1).export("tsdf_grid_in_cam_coords.ply")
    vec2 = np.matmul(depth_intrinsic[:3, :3], vec1.T)
    image_coords = vec2
    image_coords[0, :] = (image_coords[0, :] /
                          image_coords[2, :])  # .astype(int)
    image_coords[1, :] = (image_coords[1, :] /
                          image_coords[2, :])  # .astype(int)
    z_mask = image_coords[2, :] > 0
    image_in_mask = (z_mask) & (image_coords[0, :] >= 0) & (
        image_coords[1, :] >= 0) & (image_coords[0, :] < dw) & (image_coords[1, :] < dh)
    # print(image_in_mask.sum())
    inside_img_inds = np.nonzero(image_in_mask)
    inds = image_coords[:, image_in_mask[:]].astype(int)
    flat_depth = depth_image.flatten()

    xs = grid_inds[image_in_mask][:, 0]
    ys = grid_inds[image_in_mask][:, 1]
    zs = grid_inds[image_in_mask][:, 2]

#     inds1d = (xs * grid_shape[1] + ys) * grid_shape[2] + zs;
#     mapping[inds1d] = inds[1,:]*dw + inds[0,:]
    depths = flat_depth[inds[1, :]*dw + inds[0, :]]
    depth_inds = np.nonzero(depths == 0)
    depths = depths[depths > 0]
    image_in_mask[np.nonzero(image_in_mask)[0][depth_inds]] = False
    homo = np.ones_like(image_coords[:, image_in_mask])
    homo[0, :] = image_coords[:, image_in_mask][0, :]
    homo[1, :] = image_coords[:, image_in_mask][1, :]

    homogenImagePositions = np.matmul(
        np.linalg.inv(depth_intrinsic[:3, :3]), homo)
    lmbds = np.linalg.norm(homogenImagePositions, axis=0)
    # lmbds
    res = vec1[image_in_mask, :]
    res[:, 0] *= 1/lmbds
    res[:, 1] *= 1/lmbds
    res[:, 2] *= 1/lmbds

    # maybe we need transformation
    values = -1 * (np.linalg.norm(-res, axis=1) - depths)

    sdfValues = values/TRUNCATION

    sdfValues[(sdfValues > 1) & (sdfValues != 0)] = 1
    sdfValues[(sdfValues < -1) & (sdfValues != 0)] = -1

    xs = grid_inds[image_in_mask][:, 0]
    ys = grid_inds[image_in_mask][:, 1]
    zs = grid_inds[image_in_mask][:, 2]

    inds1d = (xs * grid_shape[1] + ys) * grid_shape[2] + zs

    grid[inds1d] = sdfValues
    return grid.reshape(grid_shape)


def project_rgb_to_depth(curr_rgb_img, color_intrinsic, depth_intrinsic, depth_img, colorToDepthExtr):
    """[Projects rgb image to depth image using color intrinsics]

    Args:
        curr_rgb_img ([type]): [description]
        color_intrinsic ([type]): [description]
        depth_intrinsic ([type]): [description]
        depth_img ([type]): [description]
        colorToDepthExtr ([type]): [description]

    Returns:
        [type]: [description]
    """
    rgb_dh, rgb_dw, _ = curr_rgb_img.shape
    dh, dw = depth_img.shape
    rgb_image_inds = np.indices([rgb_dh, rgb_dw]).reshape(2, -1).T
    new_img = np.zeros([dh, dw, 3])
    color_intrinsic_inv = np.linalg.inv(color_intrinsic[:3, :3])
    img_coords_stacked = np.hstack(
        (rgb_image_inds, np.ones([rgb_image_inds.shape[0], 1]))).astype(int)
    img_coords_rgb_cam = np.matmul(color_intrinsic_inv, img_coords_stacked.T)
    img_coords_depth_cam = np.matmul(
        colorToDepthExtr[:3, :3], img_coords_rgb_cam).T + colorToDepthExtr[:-1, 3]
    img_coords_depth_img = np.matmul(
        depth_intrinsic[:3, :3], img_coords_depth_cam.T)
    image_in_mask = (img_coords_depth_img[0, :] >= 0) & (img_coords_depth_img[1, :] >= 0) & (
        img_coords_depth_img[0, :] < dh) & (img_coords_depth_img[1, :] < dw)

    depth_inds = img_coords_depth_img.T[image_in_mask][:, :2].astype(int)
    color_inds = rgb_image_inds[image_in_mask]
    new_img[depth_inds[:, 0], depth_inds[:, 1],
            0] = curr_rgb_img[color_inds[:, 0], color_inds[:, 1], 0]
    new_img[depth_inds[:, 0], depth_inds[:, 1],
            1] = curr_rgb_img[color_inds[:, 0], color_inds[:, 1], 1]
    new_img[depth_inds[:, 0], depth_inds[:, 1],
            2] = curr_rgb_img[color_inds[:, 0], color_inds[:, 1], 2]
    return new_img


def represents_int(s):
    ''' if string s represents an int. '''
    try:
        int(s)
        return True
    except ValueError:
        return False


def read_label_mapping(filename, label_from='raw_category', label_to='nyu40id'):
    """utility for scannet data reading

    Args:
        filename ([type]): [description]
        label_from (str, optional): [description]. Defaults to 'raw_category'.
        label_to (str, optional): [description]. Defaults to 'nyu40id'.

    Returns:
        [type]: [description]
    """
    assert os.path.isfile(filename)
    mapping = dict()
    with open(filename) as csvfile:
        reader = csv.DictReader(csvfile, delimiter='\t')
        for row in reader:
            mapping[row[label_from]] = int(row[label_to])
    if represents_int(list(mapping.keys())[0]):
        mapping = {int(k): v for k, v in mapping.items()}
    return mapping


def read_segmentation(filename):
    """[Reads segmentation information for given scene]

    Args:
        filename ([type]): [description]

    Returns:
        [type]: [description]
    """
    assert os.path.isfile(filename)
    seg_to_verts = {}
    with open(filename) as f:
        data = json.load(f)
        num_verts = len(data['segIndices'])
        for i in range(num_verts):
            seg_id = data['segIndices'][i]
            if seg_id in seg_to_verts:
                seg_to_verts[seg_id].append(i)
            else:
                seg_to_verts[seg_id] = [i]
    return seg_to_verts, num_verts


def read_aggregation(filename):
    """Reads aggregation file. From Votenet

    Args:
        filename ([type]): [description]

    Returns:
        [type]: [description]
    """
    assert os.path.isfile(filename)
    object_id_to_segs = {}
    label_to_segs = {}
    with open(filename) as f:
        data = json.load(f)
        num_objects = len(data['segGroups'])
        for i in range(num_objects):
            object_id = data['segGroups'][i]['objectId'] + \
                1  # instance ids should be 1-indexed
            label = data['segGroups'][i]['label']
            segs = data['segGroups'][i]['segments']
            object_id_to_segs[object_id] = segs
            if label in label_to_segs:
                label_to_segs[label].extend(segs)
            else:
                label_to_segs[label] = segs
    return object_id_to_segs, label_to_segs


def read_mesh_vertices(filename):
    """ read XYZ for each vertex.
    """
    assert os.path.isfile(filename)
    with open(filename, 'rb') as f:
        plydata = PlyData.read(f)
        num_verts = plydata['vertex'].count
        vertices = np.zeros(shape=[num_verts, 3], dtype=np.float32)
        vertices[:, 0] = plydata['vertex'].data['x']
        vertices[:, 1] = plydata['vertex'].data['y']
        vertices[:, 2] = plydata['vertex'].data['z']
    return vertices


def read_mesh_vertices_rgb(filename):
    """ read XYZ RGB for each vertex.
    Note: RGB values are in 0-255
    """
    assert os.path.isfile(filename)
    with open(filename, 'rb') as f:
        plydata = PlyData.read(f)
        num_verts = plydata['vertex'].count
        vertices = np.zeros(shape=[num_verts, 6], dtype=np.float32)
        vertices[:, 0] = plydata['vertex'].data['x']
        vertices[:, 1] = plydata['vertex'].data['y']
        vertices[:, 2] = plydata['vertex'].data['z']
        vertices[:, 3] = plydata['vertex'].data['red']
        vertices[:, 4] = plydata['vertex'].data['green']
        vertices[:, 5] = plydata['vertex'].data['blue']
    return vertices


def read_scan_properties(scan_path, scene_name, label_map, axis_align_matrix):
    """reads all the info for desired scan

    Args:
        scan_path ([type]): [description]
        scene_name ([type]): [description]
        label_map ([type]): [description]
        axis_align_matrix ([type]): [description]
    """
    scene_mesh = read_mesh_vertices_rgb(os.path.join(
        scan_path, "{}_vh_clean_2.ply".format(scene_name)))
    colors = scene_mesh[:, 3:]
    scene_mesh = trimesh.points.PointCloud(scene_mesh[:, :3])

    scene_mesh.apply_transform(axis_align_matrix)
#     scene_mesh.export("{}.ply".format(scene_name))
    scene_mesh = scene_mesh.vertices
    object_id_to_segs, label_to_segs = read_aggregation(os.path.join(
        scan_path, "{}_vh_clean.aggregation.json".format(scene_name)))
    seg_to_verts, num_verts = read_segmentation(os.path.join(
        scan_path, "{}_vh_clean_2.0.010000.segs.json".format(scene_name)))
    label_ids = np.zeros(shape=(num_verts), dtype=np.uint32)  # 0: unannotated
    object_id_to_label_id = {}
    for label, segs in label_to_segs.items():
        label_id = label_map[label]
        for seg in segs:
            verts = seg_to_verts[seg]
            label_ids[verts] = label_id
    instance_ids = np.zeros(
        shape=(num_verts), dtype=np.uint32)  # 0: unannotated
    num_instances = len(np.unique(list(object_id_to_segs.keys())))
    for object_id, segs in object_id_to_segs.items():
        for seg in segs:
            verts = seg_to_verts[seg]
            instance_ids[verts] = object_id
            if object_id not in object_id_to_label_id:
                object_id_to_label_id[object_id] = label_ids[verts][0]
    return instance_ids, label_ids, object_id_to_segs, scene_mesh, object_id_to_label_id, colors


def get_frustrum(intrinsics, dw, dh, pose_matrix, axis_align_matrix):
    """Returns camera frustum given image dimensions and camera pose

    Args:
        intrinsics ([type]): [description]
        dw ([type]): [description]
        dh ([type]): [description]
        pose_matrix ([type]): [description]
        axis_align_matrix ([type]): [description]

    Returns:
        [type]: [description]
    """
    d = 3
    frustrum = []
    frustrum.append(np.matmul(intrinsics[:, :3], np.array([0, 0, 0])))

    frustrum.append(np.matmul(intrinsics[:3, :3], np.array([0, 0, d])))
    frustrum.append(np.matmul(intrinsics[:3, :3], np.array([0, dh * d, d])))
    frustrum.append(np.matmul(intrinsics[:3, :3], np.array([dw*d, dh*d, d])))
    frustrum.append(np.matmul(intrinsics[:3, :3], np.array([dw * d, 0, d])))
    frustrum = trimesh.points.PointCloud(np.array(frustrum))
    frustrum.apply_transform(pose_matrix)
    frustrum.apply_transform(axis_align_matrix)
    return frustrum
