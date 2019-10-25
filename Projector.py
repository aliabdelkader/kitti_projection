
import numpy as np

class KittiProjector:
    """
    class to project kitti velodyne points 
    """


    ## Spherical Projection ###############################################################

    @staticmethod
    def cartesian_to_spherical(x: np.array, y: np.array, z: np.array) -> [np.array, np.array, np.array]:
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
        elevation = np.arctan2(z, depth)

        # transforms to degrees
        azimuth = np.rad2deg(azimuth)
        elevation = np.rad2deg(elevation)

        return azimuth, elevation, depth

    @staticmethod
    def get_pgm_index(
                    lidar_data: np.array, 
                    pgm_height: int = 64, 
                    pgm_width: int = 512,
                    vertical_field_view: tuple = (-25, 3),
                    horizontal_field_view: tuple = (-90, 90),
                    invert_z_axis=True) -> np.array:
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
        azimuth, elevation, depth = KittiProjector.cartesian_to_spherical(x, y, z)

        vertical_step = (abs(vertical_field_view[0]) + abs(vertical_field_view[1])) / (pgm_height-1)
        horizontal_step = (abs(horizontal_field_view[0]) + abs(horizontal_field_view[1])) / (pgm_width-1)

        vertical_range = np.arange(vertical_field_view[0], vertical_field_view[1], vertical_step)
        horizontal_range = np.arange(horizontal_field_view[0], horizontal_field_view[1], horizontal_step)

        elevation_diff = np.abs(elevation.reshape(-1, 1) - vertical_range.reshape(1, -1))
        azimuth_diff = np.abs(azimuth.reshape(-1, 1) - horizontal_range.reshape(1, -1))

        pgm_azimuth_idx = np.argmin(azimuth_diff, axis=1).squeeze()
        pgm_elevation_idx = np.argmin(elevation_diff, axis=1).squeeze()

        pgm_azimuth_angles = horizontal_range[pgm_azimuth_idx]
        pgm_elevation_angles = vertical_range[pgm_elevation_idx]

        return depth, pgm_azimuth_angles, pgm_elevation_angles

   ## project points to image coordinates #############################################################

    @staticmethod
    def project_velo2cam(lidar_data: np.array, p_rect_02_matrix: np.array, r_rect_00_matrix: np.array, tr_velo_matrix: np.array) -> np.array:
        """
        function project lidar points into image
        :param
            lidar_data: lidar points to be projected, shape: number_of_points,3 (-1,x,y,z)
            p_rect_02_matrix: projection matrix
            r_rect_00_matrix: rectifying rotation matrix of cam 0 in homogeneous space
            tr_velo_matrix:   translation rotation matrix of cam 0 in homogeneous space  
        :return:
            projection_points_normalized: projected points in pixel coordinates: u,v
        """

        # transform lidar points to homogeneous space (x,y,z,1)
        velo_points_homogenous = np.concatenate([lidar_data, np.ones((lidar_data.shape[0], 1))], axis=1)

        # calculate projection matrix:
        projection_matrix = np.dot(p_rect_02_matrix, np.dot(r_rect_00_matrix, tr_velo_matrix))

        # project points [ x,y,z,1] --> [x,y,s]
        projection_points = np.dot(projection_matrix, velo_points_homogenous.T)

        # normalize projected points [x,y,s] --> [x/s,y/s] = [u,v]
        projection_points_normalized = projection_points[:2, :] / projection_points[2, :]

        return projection_points_normalized.T