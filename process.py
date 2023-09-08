import SimpleITK
import numpy as np
import torch
import torch.nn.functional as F
import glob
from pathlib import Path




class Zerofield():  
    def __init__(self):
        self.in_path = Path('/input/images')
        self.out_path = Path('/output/images/displacement-field')
        ##create displacement output folder 
        self.out_path.mkdir(parents=True, exist_ok=True)



    def load_inputs(self):


        ###print list of available files
        print('Available files:')
        print('Fixed','\t\t', True if len(list((self.in_path / 'fixed').glob('*.mha'))) == 1 else False)
        print('Moving','\t\t', True if len(list((self.in_path / 'moving').glob('*.mha'))) == 1 else False)
        print('Fixed Mask','\t', True if len(list((self.in_path / 'fixed-mask').glob('*.mha'))) == 1 else False)
        print('Moving Mask','\t', True if len(list((self.in_path / 'moving-mask').glob('*.mha'))) == 1 else False)

         ## Grand Challenge Algorithms expect only one file in each input folder, i.e.:
        fpath_fixed_image = list((self.in_path / 'fixed').glob('*.mha'))[0]
        fpath_moving_image = list((self.in_path / 'moving').glob('*.mha'))[0]
        fixed_image = torch.from_numpy(SimpleITK.GetArrayFromImage(SimpleITK.ReadImage(fpath_fixed_image))).unsqueeze(0)
        moving_image = torch.from_numpy(SimpleITK.GetArrayFromImage(SimpleITK.ReadImage(fpath_moving_image))).unsqueeze(0)

        if len(list((self.in_path / 'fixed-mask').glob('*.mha'))) == 1:
            fixed_mask = torch.from_numpy(SimpleITK.GetArrayFromImage(SimpleITK.ReadImage(list((self.in_path / 'fixed-mask').glob('*.mha'))[0]))).unsqueeze(0)
        else:
            fixed_mask = None
        if len(list((self.in_path / 'moving-mask').glob('*.mha'))) == 1:
            moving_mask = torch.from_numpy(SimpleITK.GetArrayFromImage(SimpleITK.ReadImage(list((self.in_path / 'moving-mask').glob('*.mha'))[0]))).unsqueeze(0)
        else:
            moving_mask = None

        return fixed_image, moving_image, fixed_mask, moving_mask

    def write_outputs(self, outputs):
        out = SimpleITK.GetImageFromArray(outputs)
        ##You can give the output-mha file any name you want, but it must be in the /output/images/displacement-field folder
        SimpleITK.WriteImage(out, str(self.out_path / 'thisIsAnArbitraryFilename.mha'))
        return 
    
    def predict(self, inputs):
        # Read the input images
        moving_image, fixed_image, moving_mask, fixed_mask = inputs
        D, H, W = fixed_image.shape[1:]

        displacement_field = torch.zeros((D, H, W, 3), dtype=torch.float32)
        return displacement_field

    def process(self):
        inputs = self.load_inputs()
        outputs = self.predict(inputs)
        self.write_outputs(outputs)

if __name__ == "__main__":
    Zerofield().process()
