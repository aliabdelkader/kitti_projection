import os
import scipy.io
import numpy as np
import pykitti
import glob
from tqdm import tqdm

import matplotlib.pyplot as plt

zhang_base_dir = "/home/fusionresearch/Datasets/KITTI_Zhang"  # directory of annotation files
kitti_tracking_dir = "/home/fusionresearch/Datasets/Kitti_Tracking_2012"  # directory of kitti tracking data
calib_dir = '/home/fusionresearch/Datasets/Kitti_Tracking_2012/data_tracking_calib/training/calib'  # calibration file
output_dir = "."

TRANSLATION_ROTATION_MATRIX = 'Tr_velo_cam'
TRANSLATION_ROTATION_SHAPE = (3, 4)
ROTATION_RECT = 'R_rect'
ROTATION_RECT_SHAPE = (3, 3)
P_RECT = 'P{cam}:'
P_RECT_SHAPE = (3, 4)

pgm_height=64
pgm_width=512

def get_list_kitti_annotation_files(root_dir):
    """
    function returns list of annotation files
    :param
        root_dir: direction to search for .mat files in, zhang dataset has annotation files in this format
    :return
       found_files: list of found annotation files,
                    each item in this list is a tuple consisting of
                            train_val: string indicating if file belongs to training or validation
                            sequence_number: sequence number that annotation file is for
                            file_id: id of file that annotation file is ground truth fo
                            data: annotation data, ground truth
    """
    found_files = []
    for root, dirs, files in os.walk(root_dir, topdown=True):
        for name in files:
            if '.mat' in name:
                dire , sequence_number = os.path.split(root)
                _, train_val = os.path.split(dire)
                file_id = name.replace('.mat', "")
                mat = scipy.io.loadmat(os.path.join(root, name))
                data = mat['truth']
                found_files.append((train_val, sequence_number, file_id, data))
    return found_files


def read_data(kitti_tracking_dir, annotation_files):
    """
    function read data, lidar scan and images for each file in list of annotation_files
    assumes that annotation files are in form returned by get_list_kitti_annotation_files function
    assumes that every annotation file has a corresponding data in kitti_tracking_dir
    assumes dataset used in kitti tracking dataset
    :param
        kitti_tracking_dir: directory that has kitti tracking data: lidar scans, images
        annotation_files: list of annotation files to read the corresponding data of
    :return
         dataset: list of dictionary items for every annotation file
                    dictionary items consists of:
                       folder_type: string to indicate with data is for training or testing
                       seq_number: sequence number of this data
                       frame_id: id of frame of data
                       ground_truth: annotation of data, labels
                       image: image of seq_number and frame_id
                       velo_data: lidar scan of seq_number and frame_id
    """
    dataset = []
    for train_val,seq_number,frame_id,ground_truth in annotation_files:
        #To-do: make it generic, reading from both from training / testing tracking dataset folders
        basedir = os.path.join(kitti_tracking_dir,'training')
        if not (os.path.isdir(basedir)):
            continue
        loaded = pykitti.tracking(basedir,seq_number)
        if not loaded.velo_files:
            print("cannot be loaded",basedir,seq_number)
            continue
        image = loaded.get_cam2(int(frame_id) )
        image = np.array(image)
        velo_data = loaded.get_velo(int(frame_id))
        if velo_data.size == 0:
            print("data could be get",seq_number,frame_id)
            continue
        dataset.append({
            'folder_type': train_val,
            'seq_number': seq_number,
            'frame_id': frame_id,
            'ground_truth': ground_truth,
            'velo_data': velo_data,
            'image': image
        })
    return dataset


def get_file_content(file_path):
    """ function read content of file in file_path """
    with open(file_path, 'r') as f:
        content = f.readlines()
    return content

###################################################################### project points to image #############################################################


def get_matrix_from_file_content(content, matrix_id):
    """
    function get matrix from content of file
    :param
        content: list of lines read from  text file that has the matrix
        matrix_id: id of matrix to retrevie from content
    :returns matrix if found
    example:
        to get translation_rotation matrix from content
        get_matrix_from_file_content(content, "Tr_velo_cam")
    """
    for line in content:
        line_split = line.split(' ')
        if line_split[0] == matrix_id:
            l = ' '.join(line_split[1:])
            return np.fromstring(l, sep=' ')
    return None


def get_translation_rotation_Velo_matrix(calib_file, return_homogeneous=True):
    """
    function get translation rotation matrix from calibration file
    :param
        calib_file: path of calibration file
        return_homogeneous: return matrix in homogeneous space
    :return:
        return matrix if found
    """
    # read calibration file content
    content = get_file_content(calib_file)
    # get translation rotation matrix from content
    tr_velo_matrix = get_matrix_from_file_content(content, TRANSLATION_ROTATION_MATRIX)
    # reshape matrix to expected shape
    tr_velo_matrix = tr_velo_matrix.reshape(TRANSLATION_ROTATION_SHAPE)
    if return_homogeneous:
        # transform matrix to homogeneous space
        result = np.eye(4)
        result[:TRANSLATION_ROTATION_SHAPE[0], :TRANSLATION_ROTATION_SHAPE[1]] = tr_velo_matrix
    else:
        result = tr_velo_matrix
    return result


def get_rectified_cam0_coord(calib_file, return_homogeneous=True):
    """
    function get rectifying rotation matrix from calibration file
    R_rect_xx is 3x3 rectifying rotation matrix to make image planes co-planar
    gets matrix of cam0 because in kitti cam0 is reference frame
    :param
        calib_file: path of calibration file
        return_homogeneous: return matrix in homogeneous space
    :return:
        return matrix if found

    """
    # read calibration file content
    content = get_file_content(calib_file)
    # get matrix from content
    matrix = get_matrix_from_file_content(content, ROTATION_RECT).reshape(ROTATION_RECT_SHAPE)
    if return_homogeneous:
        # transform matrix to homogeneous space
        R_rect_00_matrix = np.identity(4)
        R_rect_00_matrix[:ROTATION_RECT_SHAPE[0], :ROTATION_RECT_SHAPE[1]] = matrix
    else:
        R_rect_00_matrix = matrix
    return R_rect_00_matrix


def get_projection_rect(calib_file, mode='02'):
    """
    function get projection matrix from calibration file
    P_rect_xx is 3x4 projection matrix after rectification
    :param
        calib_file: path of calibration file
        mode: define cam matrix to return, default '02' to color cam o2
    :return:
        return matrix if found
    """
    # read calibration file content
    content = get_file_content(calib_file)
    matrix_id = P_RECT.format(cam=int(mode))
    # get matrix from content
    matrix = get_matrix_from_file_content(content, matrix_id).reshape(P_RECT_SHAPE)
    P_rect_matrix = matrix
    return P_rect_matrix


def project_velo2cam(velo_points, calib_file):
    """
    function project lidar points into image
    :param
        velo_points: lidar points to be projected, shape: number_of_points,3 (-1,x,y,z)
        calib_file:  calibration file to read transformation matrices from
    :return:
        projection_points_normalized: projected points in pixel coordinates: u,v
    """
    # get project matrix
    P_rect_02_matrix = get_projection_rect(calib_file)

    # get rectifying rotation matrix of cam 0 in homogeneous space
    R_rect_00_matrix = get_rectified_cam0_coord(calib_file, return_homogeneous=True)

    # get translation rotation matrix of cam 0 in homogeneous space
    tr_velo_matrix = get_translation_rotation_Velo_matrix(calib_file, return_homogeneous=True)

    # transform lidar points to homogeneous space (x,y,z,1)
    velo_points_homogenous = np.concatenate([velo_points, np.ones((velo_points.shape[0], 1))], axis=1)

    # calculate projection matrix:
    projection_matrix = np.dot(P_rect_02_matrix, np.dot(R_rect_00_matrix, tr_velo_matrix))

    # project points [ x,y,z,1] --> [x,y,s]
    projection_points = np.dot(projection_matrix, velo_points_homogenous.T)

    # normalize projected points [x,y,s] --> [x/s,y/s] = [u,v]
    projection_points_normalized = projection_points[:2, :] / projection_points[2, :]

    return projection_points_normalized.T


def set_rgb_data(frame, img):
    """
    function sets the rgb values of every lidar point in frame based its pixel coordinates in img
    assumes columns 3,4 have pixel coordinates of the points
    assumes columns 5,6,7 are rgb values to be set
    :param
        frame:  lidar points to color, set rgb values for, shape:  x,y,z,u,v,r,g,b,label
        img: color image of the frame
    :return:
        frame: with rgb values set
    """
    for i in range(frame.shape[0]):
        # get pixel coordinates of point
        x, y = np.floor(frame[i, 3:5]).astype(np.int32)
        # set rgb value according to img
        frame[i, 5:8] = img[y, x].astype(np.uint8)
    return frame


def filter_points(points, image_width, image_height):
    """
    function finds indexes of points that are within image frame ( within image width and height )
    searches for
        points with x coordinate greater than zero, less than image_width
        points with y coordinate greater than zero, less than image_height
    :param
        points: points to be filter, shape: number_points,2
        image_width: width of image frame
        image_height: height of image frame
    :return:
        indexes of points that satisfy both conditions
    """
    # points with x coordinate greater than zero, less than image_width
    in_w = np.logical_and(points[:, 0] > 0, points[:, 0] < image_width)
    # points with y coordinate greater than zero, less than image_height
    in_h = np.logical_and(points[:, 1] > 0, points[:, 1] < image_height)
    return np.logical_and(in_w, in_h)



def project_dataset_to_image(dataset):
    """
    function project lidar scans to image and sets rgb value for every lidar point
    example: project lidar points to image, set their rgb values
    Args:
     dataset: list of dictionary items returned by read_data function

    return:
        list of arrays of 8 channels: x, y, z, u, v r, g, b
    """
    print("project dataset to image")
    # loop over each item in data set
    for item in tqdm(dataset):
        # folder_type = item['folder_type']
        seq_number = item['seq_number']  # sequence number
        # frame_id = item['frame_id']
        ground_truth = item['ground_truth']  # annotations
        velo_data = item['velo_data']  # lidar scan
        image = item['image']  # color image

        image_height, image_width = image.shape[0], image.shape[1]
        velo_data_xyz = velo_data[:, :3]
        # velo_data_intensity = velo_data[:, 3]

        # processed_frame: xyz uv rgb
        # index            012 34 567
        processed_frame = np.zeros((velo_data.shape[0], 8)).astype(np.float64)
        processed_frame[:, :3] = velo_data_xyz
        # processed_frame[:, -1] = prepare_labels(ground_truth).reshape(-1)
        calib_file = os.path.join(calib_dir, '{num}.txt'.format(num=seq_number))

        # remove points behind image plane
        velo_points_plane_ind = processed_frame[:, 0] > 1
        processed_frame = processed_frame[velo_points_plane_ind, :]

        # project points to image
        processed_frame[:, 3:5] = project_velo2cam(processed_frame[:, :3], calib_file)

        # remove points outside image frame
        inds = filter_points(processed_frame[:, 3:5], image_width=image_width, image_height=image_height)
        processed_frame = processed_frame[inds, :]

        # set rgb of remaining points
        processed_frame = set_rgb_data(processed_frame, image)

        item['lidar_rgb'] = processed_frame.copy()

        del processed_frame

    return dataset


################################################ Spherical Projection ###############################################################

def cartesian_to_spherical(x, y, z):
    """
    function transforms cartesian coordinates into spherical ones

    Args:
        x: x axis values
        y: y axis values
        z: z axis values

    returns:
        spherical coordinate of every point in degrees

    """

    depth = np.sqrt(x ** 2 + y ** 2 + z ** 2)

    azimuth = np.arctan2(y, x)
    elevation = np.arcsin(z / depth)

    # transforms to degrees
    azimuth = np.rad2deg(azimuth)
    elevation = np.rad2deg(elevation)

    return azimuth, elevation, depth


def shift_range(x, new_min=0., new_max=1., old_min=None, old_max=None):
    """
    function shift values in array from an old range to new range
    Args:
        x: input array
        new_min: min value of new range
        new_max: max value of new range
        old_min: min value of input range
        old_max: max value of input range
    Returns:
        normalize array
    """
    if old_max and old_min:
        new_value = (new_max - new_min) / (old_max - old_min) * (x - old_max) + new_max
    else:
        new_value = (new_max - new_min) / (x.max() - x.min()) * (x - x.max()) + new_max
    return new_value


def get_pgm_index(lidar_data, pgm_height, pgm_width, vertical_field_view, horizontal_field_view, invert_z_axis=True):
    """
    function get polar grid map coordinates for every point in lidar scan
    Args:
        lidar_data: numpy array of lidar scan
        pgm_height: height of polar grid map
        pgm_width: width of polar grid map
        vertical_field_view:  tuple
            vertical field of view of lidar, angle of highest layer, angle of lowest layer
            for sick: (-1.5,1.5)
            for velodyne 64: (-25, 3)
        horizontal_field_view: horizontal field of view of layer, angles of extrem laser beans
            for sick (-42.5, 42.5)
            for velodyne 64: (-180, 180), for kitti (-90, 90)
        invert_z_axis: boolean, flag wether to invert z axis

    return:
       pgm_azimuth_idx, pgm_elevation_idx, depth: angles for every lidar point and its depth
    """

    # extract coordinate values
    x, y, z = lidar_data[:, 0], lidar_data[:, 1], lidar_data[:, 2]

    if invert_z_axis:
        z = -1 * z

    # get spherical coordinates of points
    azimuth, elevation, depth = cartesian_to_spherical(x, y, z)

    azimuth = shift_range(azimuth, new_min=vertical_field_view[0], new_max=vertical_field_view[1])
    elevation = shift_range(elevation, new_min=horizontal_field_view[0], new_max=horizontal_field_view[1])

    vertical_step = (abs(vertical_field_view[0]) + abs(vertical_field_view[1])) / pgm_height
    horizontal_step = (abs(horizontal_field_view[0]) + abs(horizontal_field_view[1])) / pgm_width

    vertical_range = np.arange(vertical_field_view[0], vertical_field_view[1], vertical_step)
    horizontal_range = np.arange(horizontal_field_view[0], horizontal_field_view[1], horizontal_step)

    elevation_diff = np.abs(elevation.reshape(-1, 1) - vertical_range.reshape(1, -1))
    azimuth_diff = np.abs(azimuth.reshape(-1, 1) - horizontal_range.reshape(1, -1))

    pgm_azimuth_idx = np.argmin(azimuth_diff, axis=1).squeeze()
    pgm_elevation_idx = np.argmin(elevation_diff, axis=1).squeeze()

    return pgm_azimuth_idx, pgm_elevation_idx, depth

def spherical_project_dataset(dataset):
    for item in tqdm(dataset):
        # folder_type = item['folder_type']
        seq_number = item['seq_number']  # sequence number
        # frame_id = item['frame_id']
        ground_truth = item['ground_truth']  # annotations
        velo_data = item['velo_data']  # lidar scan

        xyz, intensity = velo_data[:, :3], velo_data[:, 3]

        lidar_rgb = item['lidar_rgb']

        rgb = lidar_rgb[:, 5:]

        pgm_azimuth, pgm_elevation, depth = get_pgm_index(lidar_data=xyz,
                                                            pgm_height=pgm_height,
                                                            pgm_width=pgm_width,
                                                            vertical_field_view=(-25, 3),
                                                            horizontal_field_view=(-90/2.0, 90/2.0),
                                                            invert_z_axis=True)

        # # result shape [ pgm_width, pgm_height, x,y,z, intensity, range, r, g, b]
        # result = np.zeros((pgm_height, pgm_width, 8), dtype=np.float)

        # for idx in range(velo_data.shape[0]):
        #     pgm_azimuth_idx = pgm_azimuth[idx]
        #     pgm_elevation_idx = pgm_elevation[idx]
        #     result[pgm_elevation_idx, pgm_azimuth_idx, :3] = xyz[idx, :3]
        #     result[pgm_elevation_idx, pgm_azimuth_idx, 3] = intensity[idx]
        #     result[pgm_elevation_idx, pgm_azimuth_idx, 4] = depth[idx]
        #     result[pgm_elevation_idx, pgm_azimuth_idx, 5:] = rgb[idx]
        #     # result[pgm_elevation_idx, pgm_azimuth_idx, 5] = convert_ground_truth(lidar_labels_data[idx][0])
        
        # result shape [ pgm_width, pgm_height, x,y,z, intensity, range, r, g, b]
        result = np.zeros((pgm_height, pgm_width, 3), dtype=np.uint8)

        for idx in range(rgb.shape[0]):
            pgm_azimuth_idx = pgm_azimuth[idx]
            pgm_elevation_idx = pgm_elevation[idx]
            result[pgm_elevation_idx, pgm_azimuth_idx, :3] = rgb[idx].astype(np.uint8)
           
        item['pgm'] = result.copy()

        del result

    return dataset


def save_image(arr, image_name=None, gray=False):
    
    fig, ax = plt.subplots(figsize=(13, 13))
    # fig, ax = plt.subplots()
    # ax.axis("off")
    if gray:
        ax.imshow(arr, cmap='gray')
    else:
        ax.imshow(arr)
    fig.savefig(image_name)


def write_dataset_csv(dataset, data_type):
    """
    function writes dataset to disk in format of csv files, each csv file is name sequence_frame.csv
    :param dataset: list of dictionary items, each item should have following keys
                    folder_type: indicate if it is train or test file
                    seq_number: sequence number of data
                    frame_id: id of frame of data
                    pgm: data to be written in csv file, shape: number_of_points X (pgm_height, pgm_width, x, y, z, intesity, rgb, depth)
    :param data_type: key of data to write to disk, can be pgm or lidar_rgb
    """
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    print("write dataset to disk")
    for item in tqdm(dataset):
        folder_type = item['folder_type']
        seq_number = item['seq_number']
        frame_id = item['frame_id']
        data = item[data_type]
        save_dir_path = os.path.join(output_dir, folder_type)
        if not os.path.isdir(save_dir_path):
            os.makedirs(save_dir_path)
        save_file_str = '_'.join([seq_number, frame_id])
        save_file_path = os.path.join(save_dir_path, save_file_str)
        # np.savetxt(save_file_path + '.csv', data, delimiter=',')
        save_image(data, image_name=save_file_path + ".png", gray=False)
    return None


def main():

    # read lidar annotation files
    zhang_velodyne_annotations_path = os.path.join(zhang_base_dir, "velodyne")
    velodyne_annotation_files = get_list_kitti_annotation_files(zhang_velodyne_annotations_path)

    # read image annotation files
    zhang_images_annotations_path = os.path.join(zhang_base_dir, "image_02")
    images_annotation_files = get_list_kitti_annotation_files(zhang_images_annotations_path)

    assert len(velodyne_annotation_files) == len(images_annotation_files)

    # read data, lidar scans, images
    dataset = read_data(kitti_tracking_dir, velodyne_annotation_files)

    dataset = dataset[:2]

    # image data projection
    dataset = project_dataset_to_image(dataset)

    #spherical projection
    dataset = spherical_project_dataset(dataset)

    # write pgm to disk
    write_dataset_csv(dataset[:10], "pgm")

if __name__ == "__main__":
    main()