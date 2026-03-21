import os
import random
import shutil   # High-level file operations, such as copying and removing files
from itertools import islice  # for splitting lists

outputFolderPath = "Dataset/SplitData"
inputFolderPath = "Dataset/all"
splitRatio = {"train": 0.7, "val": 0.2, "test": 0.1}
classes = ["fake", "real"]

try:   # checks if dir structure have output folder path or not
    shutil.rmtree(outputFolderPath)   # Removes the directory and its contents.
except OSError as e:
    os.mkdir(outputFolderPath)

# --------  Directories to Create -----------
os.makedirs(f"{outputFolderPath}/train/images", exist_ok=True)
os.makedirs(f"{outputFolderPath}/train/labels", exist_ok=True)
os.makedirs(f"{outputFolderPath}/val/images", exist_ok=True)
os.makedirs(f"{outputFolderPath}/val/labels", exist_ok=True)
os.makedirs(f"{outputFolderPath}/test/images", exist_ok=True)
os.makedirs(f"{outputFolderPath}/test/labels", exist_ok=True)

# --------  Get the Names  -----------
listNames = os.listdir(inputFolderPath)   # give names of all files in input folder path

uniqueNames = []
for name in listNames:   # remove duplicate images take only unique
    uniqueNames.append(name.split('.')[0])
uniqueNames = list(set(uniqueNames))

# --------  Shuffle -----------
random.shuffle(uniqueNames)

# --------  Find the number of images for each folder -----------
lenData = len(uniqueNames)
lenTrain = int(lenData * splitRatio['train'])
lenVal = int(lenData * splitRatio['val'])
lenTest = int(lenData * splitRatio['test'])

# --------  Put remaining images in Training -----------
if lenData != lenTrain + lenTest + lenVal:
    remaining = lenData - (lenTrain + lenTest + lenVal)
    lenTrain += remaining

# --------  Split the list -----------
lengthToSplit = [lenTrain, lenVal, lenTest]
Input = iter(uniqueNames)
Output = [list(islice(Input, elem)) for elem in lengthToSplit]
# Splits the list of unique names into training, validation, and test sets.
print(f'Total Images:{lenData} \nSplit: {len(Output[0])} {len(Output[1])} {len(Output[2])}')

# --------  Copy the files  -----------

sequence = ['train', 'val', 'test']
for i, out in enumerate(Output):
    for fileName in out:
        shutil.copy(f'{inputFolderPath}/{fileName}.jpg', f'{outputFolderPath}/{sequence[i]}/images/{fileName}.jpg')
        shutil.copy(f'{inputFolderPath}/{fileName}.txt', f'{outputFolderPath}/{sequence[i]}/labels/{fileName}.txt')
    # Copies the file to the appropriate split folder
print("Split Process Completed...")


# -------- Creating Data.yaml file  -----------

dataYaml = f'path: ../Data\n\
train: ../train/images\n\
val: ../val/images\n\
test: ../test/images\n\
\n\
nc: {len(classes)}\n\
names: {classes}'


f = open(f"{outputFolderPath}/data.yaml", 'a')
f.write(dataYaml)
f.close()

print("Data.yaml file Created...")
