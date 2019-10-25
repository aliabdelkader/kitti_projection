from ZhangDataReader import ZhangDatasetReader
from Projector import KittiProjector
import numpy as np
from tqdm import tqdm

def filter_points(points: np.array, image_width: int, image_height: int) -> np.array:
    """
    function finds indexes of points that are within image frame ( within image width and height )
    searches for
        points with x coordinate greater than zero, less than image_width
        points with y coordinate greater than zero, less than image_height
    Args:
        points: points to be filter, shape: number_points,2
        image_width: width of image frame
        image_height: height of image frame
    return:
        indexes of points that satisfy both conditions
    """
    # points with x coordinate greater than zero, less than image_width
    in_w = np.logical_and(points[:, 0] > 0, points[:, 0] < image_width)
    # points with y coordinate greater than zero, less than image_height
    in_h = np.logical_and(points[:, 1] > 0, points[:, 1] < image_height)
    return np.logical_and(in_w, in_h)


def get_rgb(image_coordinates: np.array, image: np.array) -> np.array:
    """
    function gets rgb value from image

    Args:  
        images coordinates to get rgb values of
        image: rgb image
    Returns
        np array with rgb value for every point
    """ 

    result = np.zeros((image_coordinates.shape[0], 3), dtype=np.float64)

    for idx in range(image_coordinates.shape[0]):
        # get pixel coordinates of point
        x, y = np.floor(image_coordinates[idx, :]).astype(np.int64)

        # set rgb value according to image
        result[idx, :] = image[y, x].astype(np.float64)
    
    return result

if __name__ == "__main__":

    zhang_base_dir = "/home/fusionresearch/Datasets/KITTI_Zhang"  # directory of annotation files
    kitti_tracking_dir = "/home/fusionresearch/Datasets/Kitti_Tracking_2012"  # directory of kitti tracking data
    calib_dir = '/home/fusionresearch/Datasets/Kitti_Tracking_2012/data_tracking_calib/training/calib'  # calibration file

    zhang_dataset = ZhangDatasetReader(kitti_tracking_dir=kitti_tracking_dir, zhang_lidar_annotation_dir=zhang_base_dir, calib_root_dir=calib_dir).read_data()

    pgm_height = 64
    pgm_width = 512
    vertical_field_view = (-5, 5)
    horizontal_field_view = (-40, 40)

    result_frames = []
    for frame_dic in tqdm(zhang_dataset[:1]):

        train_val = frame_dic['folder_type']
        seq_number = frame_dic['seq_number']
        frame_id   =  frame_dic['frame_id']
        lidar_ground_truth =  frame_dic['lidar_ground_truth']
        velo_data  = frame_dic['velo_data']
        image = frame_dic['image']
        tr_velo_matrix  = frame_dic["tr_velo_matrix"]
        R_rect_00_matrix = frame_dic["R_rect_00_matrix"]
        P_rect_matrix = frame_dic["P_rect_matrix"]

        image_height, image_width = image.shape[0], image.shape[1]

        xyz, intensity = velo_data[:, :3], velo_data[:, 3]

        depth, pgm_azimuth_angles, pgm_elevation_angles = KittiProjector.get_pgm_index(
                                                                                        lidar_data= xyz, 
                                                                                        pgm_height= pgm_height, 
                                                                                        pgm_width = pgm_width,
                                                                                        vertical_field_view = vertical_field_view,
                                                                                        horizontal_field_view = horizontal_field_view,
                                                                                        invert_z_axis=True)

        
        # image coordinates of filter points
        image_coordinates = KittiProjector.project_velo2cam( lidar_data=xyz,
                                                            p_rect_02_matrix= P_rect_matrix,
                                                            r_rect_00_matrix= R_rect_00_matrix, 
                                                            tr_velo_matrix= tr_velo_matrix)
        
        # get index of points behind image plane
        velo_points_plane_ind = xyz[:, 0] > 1

        # get index of points outside image frame
        inds = filter_points(image_coordinates, image_width=image_width, image_height=image_height)

        image_filter_condition = np.logical_and(velo_points_plane_ind, inds)

        # get rgb values
        rgb = get_rgb(image_coordinates[image_filter_condition, :], image)

        # shape: x, y, z, intesnity, depth, azimuth, elevation, x_image, y_image, r, g, b
        processed_frame = np.zeros((velo_data.shape[0], 12), dtype=np.float64)

        processed_frame[:, :3] = xyz
        processed_frame[:, 3] = intensity
        processed_frame[:, 4] = depth
        processed_frame[:, 5] = pgm_elevation_angles
        processed_frame[:, 6] = pgm_azimuth_angles
        processed_frame[:, 7:9] =  image_coordinates
        processed_frame[image_filter_condition, 9:] = rgb

        # np.savetxt("{seq_num}_{frame_id}.csv".format(seq_num=seq_number, frame_id=frame_id),processed_frame, delimiter=",")

        np.save("{seq_num}_{frame_id}.npy".format(seq_num=seq_number, frame_id=frame_id), processed_frame)
        result_frames.append(processed_frame)