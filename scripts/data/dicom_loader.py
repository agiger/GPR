import os
import sys
import shutil
import pydicom


class DicomLoader():
    def __init__(self, opt):
        self.input_dir = opt.input_dir
        self.output_dir = opt.output_dir
        self.n_slices = opt.n_slices
        self.files = sorted([os.path.join(self.input_dir, i) for i in os.listdir(self.input_dir)
                             if os.path.isfile(os.path.join(self.input_dir, i))])
        self.renamed_files = []
        self.is_navi = opt.is_navi

    def preprocess(self):
        # Load DICOMs and rename files
        self.rename_files()

        if self.is_navi:
            print('Navigators: Set spacing between slices to 1')
            self.set_spacing_between_slices()
        else:
            print('Data files: Sort data slices according to their slice position')
            self.sort_dats()

    def rename_files(self):
        for i, file in enumerate(self.files):
            dcm = pydicom.read_file(self.files[i])

            if self.is_navi and ('ImageComments' in dcm and dcm.ImageComments == 'Navigator'):
                new_file = os.path.join(self.output_dir, 'navi%05d.dcm' % dcm.InstanceNumber)
            else:
                new_file = os.path.join(self.output_dir, 'data%05d.dcm' % dcm.InstanceNumber)

            shutil.copyfile(file, new_file)
            self.renamed_files.append(new_file)

    def set_spacing_between_slices(self):
        for i, file in enumerate(self.renamed_files):
            dcm = pydicom.read_file(self.renamed_files[i])

            if dcm.SpacingBetweenSlices == 0:
                dcm.SpacingBetweenSlices = 1
                dcm.save_as(self.renamed_files[i])

    def sort_dats(self):
        n_images = len(self.renamed_files)
        n_sweeps = int(n_images/self.n_slices)

        if not (n_images % self.n_slices) == 0:
            sys.exit('Number of slice positions is not correct')

        for p in range(0, self.n_slices):
            dest_dir = os.path.join(self.output_dir, 'sorted', 'slice%02d' % (p+1))
            os.makedirs(dest_dir, exist_ok=True)

            for i in range(0, n_sweeps):
                shutil.copy2(self.renamed_files[p+i*self.n_slices], dest_dir)

    def get_files(self):
        return self.files

    def get_files_renamed(self):
        return self.renamed_files
