import SimpleITK
import numpy as np
import torch
import torch.nn.functional as F
import glob
from pathlib import Path

#pip install -q "monai-weekly[nibabel, tqdm]" itk
# Installing the recommended dependencies 
# https://docs.monai.io/en/stable/installation.html


class Zerofield():  # SegmentationAlgorithm is not inherited in this class anymore
    def __init__(self):
        pass


    def load_inputs(self):
        ##

        fpath_fixed_image = glob.glob('/input/images/fixed/*.mha')[0]
        fpath_moving_image = glob.glob('/input/images/moving/*.mha')[0]
        fpath_fixed_mask = glob.glob('/input/images/fixed-mask/*.mha')[0]
        fpath_moving_mask = glob.glob('/input/images/moving-mask/*.mha')[0]
    
        fixed_image = torch.from_numpy(SimpleITK.GetArrayFromImage(SimpleITK.ReadImage(fpath_fixed_image))).unsqueeze(0)
        ##read other stuff
        moving_image = torch.from_numpy(SimpleITK.GetArrayFromImage(SimpleITK.ReadImage(fpath_moving_image))).unsqueeze(0)
        fixed_mask = torch.from_numpy(SimpleITK.GetArrayFromImage(SimpleITK.ReadImage(fpath_fixed_mask))).unsqueeze(0)
        moving_mask = torch.from_numpy(SimpleITK.GetArrayFromImage(SimpleITK.ReadImage(fpath_moving_mask))).unsqueeze(0)
        

        return fixed_image, moving_image, fixed_mask, moving_mask

    def write_outputs(self, outputs):
        out = SimpleITK.GetImageFromArray(outputs)
        out_path = Path('/output/displacement-field/field.mha')
        out_path.parent.mkdir(parents=True, exist_ok=True)
        SimpleITK.WriteImage(out, str(out_path))
        return 
    
    def predict(self, inputs):
        # Read the input images
        moving_image, fixed_image, moving_mask, fixed_mask = inputs
        D, H, W = fixed_image.shape[1:]

        displacement_field = torch.zeros((D, H, W, 3), dtype=torch.float32)
        return displacement_field

    def process(self):
        """
        Read inputs from /input, process with your algorithm and write to /output
        """
        inputs = self.load_inputs()
        outputs = self.predict(inputs)
        self.write_outputs(outputs)

if __name__ == "__main__":
    Zerofield().process()
