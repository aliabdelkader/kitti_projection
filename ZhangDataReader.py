import scipy.io
import numpy as np
import pykitti
from pathlib import Path
from glob import glob
from tqdm import tqdm
from DataReader import KittiDatasetReader
class ZhangDatasetReader(KittiDatasetReader):
    """
    Reader Class for zhang dataset
    requires kitti tracking data, zhang annotations
    """
    TRANSLATION_ROTATION_MATRIX = 'Tr_velo_cam'
    TRANSLATION_ROTATION_SHAPE = (3, 4)
    ROTATION_RECT = 'R_rect'
    ROTATION_RECT_SHAPE = (3, 3)
    P_RECT = 'P{cam}:'
    P_RECT_SHAPE = (3, 4)
    
    def __init__(self, kitti_tracking_dir : str, calib_root_dir : str, zhang_lidar_annotation_dir : str):
        """
        Constructor of data reader

        Args:
            kitti_tracking_dir: path to root directory of kitti tracking dataset
            zhang_lidar_annotation_dir: path to root directory of lidar annotation of zhang
            calib_root_dir: path to root directory of calibration files 
        
        annotation_files: 
            list represent zhang annotation files
            each element in the list is a dictionary reresenting a frame where:
                train_val: does frame belong to zhang trainig or validation set
                sequence_number: number of the sequence to which frame belongs
                frame_id: id of the frame
                lidar_ground_truth: zhang annoatation of lidar data
        dataset: 
            list represent zhang dataset
            each element in the list is a dictionary reresenting a frame where:
                train_val: does frame belong to zhang trainig or validation set
                sequence_number: number of the sequence to which frame belongs
                frame_id: id of the frame
                lidar_ground_truth: zhang lidar annotation
                image: rgb image of the frame
                calibration matrixes
        """
        super(ZhangDatasetReader, self).__init__()
        self.kitti_tracking_dir = Path(kitti_tracking_dir)
        self.zhang_lidar_annotation_dir = Path(zhang_lidar_annotation_dir)
        self.calib_root_dir = Path(calib_root_dir)
        self.lidar_annotation_files = []
        self.dataset = []
    
    def get_lidar_annotation_files(self, files_extension : str = ".mat"):
        """
        function read annotation data from lidar annotation root directory

        Args:
            files_extension: str expected extension of annotation files
            zhang annotation extension is .mat

        """

        annotation_files_list = self.zhang_lidar_annotation_dir.rglob('*' + files_extension)

        for annotation_file_path in annotation_files_list:
            frame_id = annotation_file_path.stem
            sequence_number = annotation_file_path.parent.stem
            train_val = annotation_file_path.parent.parent.stem
            mat = scipy.io.loadmat(str(annotation_file_path))
            annotation_data = mat['truth']
            self.lidar_annotation_files.append({
                "train_val": train_val,
                "sequence_number": sequence_number,
                "frame_id": frame_id,
                "lidar_ground_truth": annotation_data,
            })
        return self
        

    def read_data(self):
        """
        function read data, lidar scan and images for each file in list of annotation_files
        assumes that annotation files are in form returned by get_annoatation_files function
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
        # read zhang annotation files
        self.get_lidar_annotation_files()
        for annotation_file in tqdm(self.lidar_annotation_files):
          

            train_val = annotation_file['train_val']
            seq_number = annotation_file["sequence_number"]
            frame_id = annotation_file["frame_id"]
            lidar_ground_truth = annotation_file["lidar_ground_truth"]

            #To-do: make it generic, reading from both from training / testing tracking dataset folders
            # basedir = os.path.join(kitti_tracking_dir,'training')
            basedir = self.kitti_tracking_dir / 'training'
            if not basedir.exists():
                print("kitti tracking directory does not exist {}".format(str(basedir)))
                break
            loaded = pykitti.tracking(str(basedir), seq_number)
            if not loaded.velo_files:
                print("cannot be loaded",str(basedir),seq_number)
                continue
            image = loaded.get_cam2(int(frame_id) )
            image = np.array(image)
            velo_data = loaded.get_velo(int(frame_id))

            tr_velo_matrix = self.get_translation_rotation_Velo_matrix(seq_number)
            R_rect_00_matrix = self.get_rectified_cam0_coord(seq_number)
            P_rect_matrix = self.get_projection_rect(seq_number, mode='02')


            if velo_data.size == 0:
                print("data could be get",seq_number,frame_id)
                continue
            self.dataset.append({
                'folder_type': train_val,
                'seq_number': seq_number,
                'frame_id': frame_id,
                'lidar_ground_truth': lidar_ground_truth,
                'velo_data': velo_data,
                'image': image,
                "tr_velo_matrix": tr_velo_matrix,
                "R_rect_00_matrix": R_rect_00_matrix,
                "P_rect_matrix": P_rect_matrix,
            })
        return self.dataset


# for testing purpose
if __name__ == "__main__":

    zhang_base_dir = "/home/fusionresearch/Datasets/KITTI_Zhang"  # directory of annotation files
    kitti_tracking_dir = "/home/fusionresearch/Datasets/Kitti_Tracking_2012"  # directory of kitti tracking data
    calib_dir = '/home/fusionresearch/Datasets/Kitti_Tracking_2012/data_tracking_calib/training/calib'  # calibration file

    zhang_dataset = ZhangDatasetReader(kitti_tracking_dir=kitti_tracking_dir, zhang_lidar_annotation_dir=zhang_base_dir, calib_root_dir=calib_dir).read_data()
    print(zhang_dataset)
