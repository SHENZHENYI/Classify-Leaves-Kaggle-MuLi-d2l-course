# Classify-Leaves-Kaggle-MuLi-d2l-course
One competition held by d2l course https://courses.d2l.ai/zh-v2/. The competition is on https://www.kaggle.com/c/classify-leaves/submissions

#### My current best result is 0.98681 in private LB, and 0.98159 in public LB. It was a late submission, so it does not show me on the LeaderBoard :( It is achieved by an ensemble of 10 models, and thanks seefun for his useful tools (I used Mixup and SoftTargetCrossEntropy), which you can download from https://github.com/seefun/TorchUtils

| Tempts  | Private LB | Public LB |
| ------------- | ------------- |  ------------- |
| one of the 5 folds  | 0.98181  | 0.97818 | 
| ensemble of 5 folds  | 0.98363  | 0.97681 |
| ensemble of 5 folds and 5 my other best models | 0.98681  | 0.98159 |


The tricks that I used are the followings:

- A set of albumentations augmentations: albumentations.Resize(112, 112, interpolation=cv2.INTER_AREA), albumentations.RandomRotate90(p=0.5), albumentations.Transpose(p=0.5), albumentations.Flip(p=0.5), albumentations.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.0625, rotate_limit=45, border_mode=1, p=0.5), #randAugment(), albumentations.Normalize(), AT.ToTensorV2(), I did not use randaug because I find it actually degrades the performance on the validation set.

- Mixup augemntation with a prob of 0.1.

- Cosine Annealing with the max_lr of 3e-4 and min_lr of max_lr/20 @ a batchsize of 64. Note that I used a technique called LRfinder to find the max_lr, a method introduced in 'Cyclical Learning Rates for Training Neural Networks' by Leslie N. Smith.

- Mixed Precision Training to accelerate the training process. It could accelerate the training on Tesla T4, but not on Tesla P100 or K80. The 3 of them are all available on Colab Pro.

- Use pretrained Resnet50d from timm. During training, I followed the finetune technique introduced in d2l course: separate the learning rate between the pretrained layers and the final fully-connected layer -- the final fully-connected layer used a lr ten times larger than the pretrained layers.

- Used AdamW with weight decay of 2e-4. It performs marginally better than Adam.

- Label smoothing helps improve 1-2%.

- Train with this configuration with 50 epochs.

- Make majority voting of 5 folds of the model.

- Use more models (my 5 other best models, varying in architectures -- seresnest50 and resnest50d, using larger resolution -- 224x224) to make votes.
