
import os
import cv2
import pickle
import dill
import uuid
import zipfile
import numpy as np
from tqdm import tqdm

class ZippedDataloader:
    def __init__(self, path_to_zip, temporal_folder = '/data3fast/users/amolina/tmp/') -> None:
        os.makedirs(temporal_folder, exist_ok=True)

        with zipfile.ZipFile(path_to_zip, 'r') as zip_ref:
            zip_ref.extractall(temporal_folder)
        
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']

        self.files = []
        for root, _, files in os.walk(temporal_folder):
            for file in files:
                if os.path.splitext(file)[1].lower() in image_extensions:
                    self.files.append(os.path.join(root, file))
        self.inner_state = 0
        self.files.sort()

        for idx, path in tqdm(enumerate(self.files), total = len(self.files), desc='Moving files to tmp folder'):
            
            subpath = path.split('/')
            extension = subpath[-1].split('.')[-1]
            fname = f"{idx:09d}.{extension}" # TODO: More elegant way
            subpath[-1] = fname
            newpath = os.path.join(*subpath)
            if temporal_folder.startswith('/'):
                newpath = '/' + newpath
            try:
                
                os.rename(path, newpath)
                cv2.cvtColor(cv2.imread(newpath, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
                self.files[idx] = newpath
            except Exception as e: print(e)
        print(f"Total files updated: {len(self)}")
        
        dill.dump(self.files, open('index.pkl', 'wb'))


    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, index):
        try:
            return cv2.cvtColor(cv2.imread(self.files[index], cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        except Exception as e:
            print(e, self.files[index], index)
            # TODO: MANAGE THIS BETTER exit()
            return np.zeros((224, 224, 3), dtype=np.uint8)
    
    def __next__(self):
        
        if self.inner_state > (len(self) - 1):
            self.inner_state += 1
            return self[self.inner_state - 1]
        
        raise StopIteration

if __name__ == '__main__':
    dataloader = ZippedDataloader('/home/adri/Pictures/zipper.zip')
    for i in dataloader:
        print(i)
