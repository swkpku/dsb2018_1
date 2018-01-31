from data.dataset import Dataset, TestDataset, DSBDataset, DSBTestDataset
from utils.config import opt, dsbopt

dataset = Dataset(opt)
testdataset = TestDataset(opt)

dsbdataset = DSBDataset(dsbopt)
dsbtestdataset = DSBTestDataset(dsbopt)

img, bbox, label, scale = dataset.__getitem__(10)

print(img)
print(bbox)
print(label)
print(scale)

img, shape, bbox, label, difficult = testdataset.__getitem__(10)

print(img)
print(shape)
print(bbox)
print(label)
print(difficult)

img, bbox, label, scale = dsbdataset.__getitem__(10)

print(img)
print(bbox)
print(label)
print(scale)

img, shape, bbox, label, difficult = dsbtestdataset.__getitem__(10)

print(img)
print(shape)
print(bbox)
print(label)
print(difficult)
