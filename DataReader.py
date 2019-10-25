import scipy.io
import numpy as np

class KittiDatasetReader:
    """
    base class for Kitti DataReader

    provides common function for kitti  data readers
    """

    
    @staticmethod
    def get_file_content(file_path):
        """ function read content of file in file_path """
        with open(file_path, 'r') as f:
            content = f.readlines()
        return content

    @staticmethod
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
        
    def create_calib_file_path(self, sequence_number: str) -> str: 
        """
        function gets the path of calib file

        Args:
            sequence_number:  sequence number to create path of

        returns:
            created path
        """
        return str (self.calib_root_dir / "{num}.txt".format(num=sequence_number))

    def read_calib_file_content(self, sequence_number: str) -> list:
        """
        function reads content of calibration file

        Args:
            sequence_number: sequence number to read calibration file of
        """
        return self.get_file_content(self.create_calib_file_path(sequence_number))

    def get_translation_rotation_Velo_matrix(self, seq_number, return_homogeneous=True):
        """
        function get translation rotation matrix from calibration file
        :param
            seq_number: sequence number to read calib file of
            return_homogeneous: return matrix in homogeneous space
        :return:
            return matrix if found
        """
        # read calibration file content
        content = self.read_calib_file_content(seq_number)
        # get translation rotation matrix from content
        tr_velo_matrix = self.get_matrix_from_file_content(content, self.TRANSLATION_ROTATION_MATRIX)
        # reshape matrix to expected shape
        tr_velo_matrix = tr_velo_matrix.reshape(self.TRANSLATION_ROTATION_SHAPE)
        if return_homogeneous:
            # transform matrix to homogeneous space
            result = np.eye(4)
            result[:self.TRANSLATION_ROTATION_SHAPE[0], :self.TRANSLATION_ROTATION_SHAPE[1]] = tr_velo_matrix
        else:
            result = tr_velo_matrix
        return result


    def get_rectified_cam0_coord(self, seq_number, return_homogeneous=True):
        """
        function get rectifying rotation matrix from calibration file
        R_rect_xx is 3x3 rectifying rotation matrix to make image planes co-planar
        gets matrix of cam0 because in kitti cam0 is reference frame
        :param
            seq_number: sequence number to read calib file of
            return_homogeneous: return matrix in homogeneous space
        :return:
            return matrix if found

        """
        # read calibration file content
        content = self.read_calib_file_content(seq_number)
        # get matrix from content
        matrix = self.get_matrix_from_file_content(content, self.ROTATION_RECT).reshape(self.ROTATION_RECT_SHAPE)
        if return_homogeneous:
            # transform matrix to homogeneous space
            R_rect_00_matrix = np.identity(4)
            R_rect_00_matrix[:self.ROTATION_RECT_SHAPE[0], :self.ROTATION_RECT_SHAPE[1]] = matrix
        else:
            R_rect_00_matrix = matrix
        return R_rect_00_matrix


    def get_projection_rect(self, seq_number, mode='02'):
        """
        function get projection matrix from calibration file
        P_rect_xx is 3x4 projection matrix after rectification
        :param
            seq_number: sequence number to read calib file of
            mode: define cam matrix to return, default '02' to color cam o2
        :return:
            return matrix if found
        """
        # read calibration file content
        content = self.read_calib_file_content(seq_number)
        matrix_id = self.P_RECT.format(cam=int(mode))
        # get matrix from content
        matrix = self.get_matrix_from_file_content(content, matrix_id).reshape(self.P_RECT_SHAPE)
        P_rect_matrix = matrix
        return P_rect_matrix