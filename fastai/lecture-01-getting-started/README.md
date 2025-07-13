# Lecture 01- Getting started

- Link: https://course.fast.ai/Lessons/lesson1.html
- Youtube Link: https://www.youtube.com/watch?v=8SF_h3xF3cE
- Fastbook chapter: https://github.com/fastai/fastbook/blob/master/01_intro.ipynb
- Jupyter Notebook 101: https://www.kaggle.com/code/jhoward/jupyter-notebook-101
- Jupyter Notebook related to the first lecture: https://www.kaggle.com/code/jhoward/is-it-a-bird-creating-a-model-from-your-own-data
- Data Ethics: https://ethics.fast.ai/
- Docs and tutorials for fast ai library: https://docs.fast.ai/ 
- Pytorch image models, timm: https://timm.fast.ai/



## Notes from the lecture

- This course will be using PyTorch. Referred to a research about Pytorch vs Tensorflow in 2022
- Fast ai is a library built on top of Pytorch
- Talks about the basics of Jupyter
- Walked through the `Is It a Bird` notebook
- Talks about the importance of getting familiar with DataBlock().
```
 DataBlock(
    blocks = (ImageBlock, CategoryBlock), --> First param is explaining what kind of input do we have, second param for mentioning what kind of output/model we require to build
    get_items=get_image_files, --> Load the input files
    splitter=RandomSplitter(valid_pct=0.2, seed=42), --> This is for validation set
    get_y=parent_label, --> How do we know the label of the photos? This returns the name of parent folder
    item_tfms=[Resize(192, method='squish')] --> In what method to resize each of the images. 
 ).dataloaders(path) 
```

```
dls = DataBlock(
    blocks=(ImageBlock, CategoryBlock),
    get_items=get_image_files,
    splitter=RandomSplitter(valid_pct=0.2, seed=42),
    get_y=parent_label,
    item_tfms=[Resize(192, method='squish')]
).dataloaders(path)

dls.show_batch(max_n = 6)
```
- More on datablock: https://docs.fast.ai/data.block.html

```
is_bird, _, probs = learn.predict(PILImage.create('bird.jpg'))
print(f"This is a: {is_bird}.")
print(f"Probability it's a bird: {probs[0]:4f}")
```


### Not just for image recognition
Segmentation
```
path = untar_data(URLs.CAMVID_TINY)
dls = SegmentationDataLoaders.from_label_func(
    path, bs=8, fnames = get_image_files(path/"images"),
    label_func = lambda o: path/'labels'/f'{o.stem}_P{o.suffix}',
    codes = np/loadtxt(path/'codes/txt', dtype=str)
)

learn = unet_learner(dls, resnet34)
learn.fine_tune(8)
```

### Tabular analysis

```
from fastai.tabular.all import *
path = untar_data(URLs.ADULT_SAMPLE)

dls = TabularDataLoaders.from_csv(path/'adult.csv', path=path, y_names="salary",
cat_names = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race'],
cont_names = ['age', 'fnlwgt', 'education-num'],
procs = [Categorify, FillMissing, Normalize])

dls.show_batch()

learn = tabular_learner(dls, metrics-accuracy)
learn.fit_one_cycle(2)
```

### Collaboritive Filtering

```
from fastai.collab import *
path = untar_data(URLS.ML_SAMPLE)
dls - CollabDataLoaders.from_csv(path/'ratings.csv')

dls.show_batch()
learn = collab_learner(dls, y_range=(0.5,0.5))
learn.fine_tune(10)
```

## Examples of the tasks in different areas for which learning is helping
- NLP
- Computer Vision
- Medicine
- Biology
- Image Recognition
- Recommendation Systems
- Playing games
- Robotics
- Financial and logistical forecasting