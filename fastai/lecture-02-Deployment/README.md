# Lecture 02 - Deployment

- Lecture Link: https://course.fast.ai/Lessons/lesson2.html
- Chapter 2 Notebook: https://github.com/fastai/fastbook/blob/master/02_production.ipynb
- Chapter 2 Solutions: https://forums.fast.ai/t/fastbook-chapter-2-questionnaire-solutions-wiki/66392 
- Saving a basic FastAI model: https://www.kaggle.com/code/jhoward/saving-a-basic-fastai-model
- Colab: https://colab.research.google.com/drive/1M-mzhZdFQ2XWBSbLCuKzrmLsm0aLEYxQ?usp=sharing
- Gradio + HuggingFace Spaces Tutorial: https://www.tanishq.ai/blog/posts/2021-11-16-gradio-huggingface.html
- HuggingFace Space: https://huggingface.co/spaces
- TinyPets: https://github.com/fastai/tinypets
- TinyPets Fork: https://github.com/jph00/tinypets
- fastsetup: https://github.com/fastai/fastsetup



## Summary

Today you’ll be designing your own machine learning project, creating your own dataset, training a model using your data, and finally deploying an application on the web. We’ll be using a particular deployment target called Hugging Face Space with Gradio, and will also see how to use JavaScript to implement an interface in the browser. Deploying to other services will look very similar to the approach you’ll study in this lesson.


### Gathering Data

Search images with duckduckgo
```
search_images_ddg

ims = search_images_ddg('grizzly bear')
len(ims)
dest = 'images/grizzly.jpg'
download_url(ims[0], dest, show_progress=False)
```

To get any help about a function:
```
?verify_images
??verify_images
doc(verify_images)
```

```
bears = DataBlock(
    blocks=(ImageBlock, CategoryBlock),
    get_items=get_image_files,
    splitter=RandomSplitter(valid_pct=0.2, seed=42),
    get_y=parent_label,
    item_tfms=Resize(128)
)
```

```
dls.valid.show_batch(max_n=4, nrows=1)
```

```
bears = bears.new(item_tfms=Resize(128, ResizeMethod.Squish))
dls = bears.dataloaders(path)
dls.valid.show_batch(max_n=4, nrows=1)
```

```
bears = bears.new(item_tfms=RandomResizeCrop(128, min_scale=0.3))
dls = bears.dataloaders(path)
dls.train.show_batch(max_n=4, nrows=1, unique=True)
```

Data Augmentation
```
bears = bears.new(item_tfms=Resize(128), batch_tfms=aug_transforms(mult=2))
dls = bears.dataloaders(path)
dls.train.show_batch(max_n=8, nrows=2, unique=True)
```

## Training Your Model, and Using it

```
learn = vision_learner(dls, resnet18, metrics=error_rate)
learn.fine_tune(4)
```

```
learn.export(model.pkl)
```

confusion matrix
```
interp = ClassificationInterpretation.from_learner(learn)
interp.plot_confusion_matrix()
interp.plot_top_losses(5, nrows=1, figsize(17,4))
```

```
cleaner = ImageClassifierCleaner(learn)
cleaner
```

```
for idx in cleaner.delete(): cleaner.fns[idx].unlink()
for idx,cat in cleaner.change(): shutil.move(str(cleaner.fns[idx]), path/cat)
```

## Gradio + HuggingFace Spaces: A tutorial

After you train a machine learning model, the next thing to do is showcase it to the  world by making a demo. Currently, the easiest way to do is with Gradio, hosting on
HuggingFace Spaces(huggingface.co/new-space). With the Gradio framework deployed on Spaces, it takes <10 minutes to deploy a model. We will use a classic CNN pet classifier as an example. 

### Saving a basic fastai model
Update/create a model in Collab/Kaggle.
Export the model using `learn.export('model.pkl')`

### Dogs v Cats

```
from fastai.vision.all import *
import gradio as gr

def is_cat(x): return x[0].isupper()

im = PILImage.create('dog.jpg')
im.thumbnail((192,192))
im

learn = load_learner('model.pkl')
learn.predict(im)
```

```
categories = ('Dog', 'Cat')

def classify_image(img):
    pred,idx,prbs = learn.predict(img)
    return dict(zip(categories, map(float, probs)))
```

```
classify_image(im)
```

gradio interface
```
#| expoert
image = gr.inputs.Image(shape=(192,192))
label = gr.outputs.Label()
examples = ['dog.jpg','cat.jpg', 'dunno.jpg']

intf = gr.Interface(fn=classify_image, inputs=image, outputs=label, example=examples)
```

## Fastsetup repo
- Clone the fastai/fastsetup git repo
- Run the conda setup script by `./setup-conda.sh`
- It will install mamba as well
- Mamba lets you to install stuff
- For example, you can run `mamba install fastai`
- `mamba install -c fastchan fastai`
- Run Jupyter `jupyter notebook --no-browser`

## High level steps
- Create a HuggigngFace Space
- Try a basic interface (use git and mamba/conda)
- Try in a noteboook (Dogs vs cats/ Pet Breeds)
- Use an expoerted learner
- Use nbdev
- Try the API (GitHub Pages): Need to fork fastai github repo and update config.yml
