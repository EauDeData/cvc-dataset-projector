import os
import cv2
import pickle
import dill
import uuid
import zipfile

class ZippedDataloader:
    def __init__(self, path_to_zip, temporal_folder = './.tmp/') -> None:
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

        self_files = []
        for idx, path in enumerate(self.files):
            
            subpath = path.split('/')
            extension = subpath[-1].split('.')[-1]
            fname = f"{idx:09d}.{extension}" # TODO: More elegant way
            subpath[-1] = fname
            newpath = os.path.join(*subpath)
            os.rename(path, newpath)

            try:
                if cv2.imread(newpath, cv2.IMREAD_COLOR) is None: raise FileNotFoundError
                self_files.append(newpath)
            except:
                continue

        self.files = self_files
        dill.dump(self.files, open('index.pkl', 'wb'))


    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, index):
        try:
            return cv2.cvtColor(cv2.imread(self.files[index], cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        except Exception as e:
            print(e, self.files[index], index)
            exit()
    
    def __next__(self):
        
        if self.inner_state > (len(self) - 1):
            self.inner_state += 1
            return self[self.inner_state - 1]
        
        raise StopIteration

if __name__ == '__main__':
    dataloader = ZippedDataloader('/home/adri/Pictures/zipper.zip')
    for i in dataloader:
        print(i)