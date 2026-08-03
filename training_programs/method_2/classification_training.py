#!/usr/bin/env python3

import os
import sys
import time
import gc
import csv
import re
import logging
import numpy as np
import cv2
import datetime
import matplotlib.pyplot as plt
from collections import defaultdict
from scipy.stats import weibull_min

import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (Conv2D, MaxPooling2D, Flatten, Dense,
                                     Dropout, BatchNormalization, Activation,
                                     Add, Input, Concatenate)
from sklearn.model_selection import train_test_split


# Global settings and hyperparameters

USE_OPENMAX = True         # Toggle OpenMax on or off
FLIP_AUG_MODE = "8x"       # Options: "4x", "8x", or "none"
SHOW_AUG_SAMPLES = False   # If True, show sample augmented images
DISPLAY_IMAGES = False     # If True, display predictions on val images

# Chose between binary class model or full class model training
BINARY_CLASSIFICATION = True

# Directory structure / data params
ROOT_SAVE_DIR = 'classification_models_training_test'
BASE_DIR  = '../../data/training_data/method_2_training_data/classification_data/sorted_to_7_classes'
IMG_SIZE  = (60, 60)
VAL_SPLIT = 0.02
GLOBAL_MAX_EPOCHS = 1000
NUM_CHECKPOINTS   = 9

TRAIN_MODEL_TYPES = [
    "baseline_cnn",
    "deeper_cnn"
]

# Defines target sub-folders for data augmentation balancing.
# None = augment all images in class. [] = augment nothing.
AUGMENTATION_RULES = {
    '3_pole': None,                
    '4_pole': None,                
    'zoverlapping': None,          
    'multiple_dislocations': None,
    'faint_probably_singles':[],  
    'dislocations': [
        'tri_shape', 
        'wavey_dislocation'
    ],
    'no_dislocation': [
        'grain_edge_no_dislocation', 
        'image_artifact_no_dislocation', 
        'wavey_no_dislocation'
    ],
    'ultra_faint': None,
}

# Create a date string for use in filenames/folders
current_date = datetime.datetime.now().strftime('%Y%m%d')
os.makedirs(ROOT_SAVE_DIR, exist_ok=True)


# Checkpoint and metadata utilities

def generate_save_epochs(max_epochs, num_checkpoints):
    """Calculates evenly spaced intervals for intermediate model checkpointing."""
    if num_checkpoints < 1:
        return []
    interval = max_epochs/(num_checkpoints+1)
    out = []
    for i in range(1,num_checkpoints+1):
        ep = int(round(i*interval))
        if 0<ep<max_epochs:
            out.append(ep)
    return sorted(set(out))

CHECKPOINT_EPOCHS = generate_save_epochs(GLOBAL_MAX_EPOCHS, NUM_CHECKPOINTS)
print(f"[INFO] Max epochs={GLOBAL_MAX_EPOCHS}")
print(f"[INFO] Intermediate checkpoints={CHECKPOINT_EPOCHS}")

def get_augmentation_suffix():
    """Returns 'aug4x', 'aug8x', or 'augNone' based on FLIP_AUG_MODE configuration."""
    mode = FLIP_AUG_MODE.lower()
    if mode == "4x":
        return "aug4x"
    elif mode == "8x":
        return "aug8x"
    else:
        return "augNone"


# Data augmentation (geometric transformations)

def augment_image_flips_4x(image):
    """Generates D4 dihedral group transformations (orthogonal flips)."""
    orig = image
    flr  = np.fliplr(image)
    fud  = np.flipud(image)
    both = np.flipud(flr)
    return np.array([orig, flr, fud, both])

def flip_main_diagonal(image):
    """Transposes tensor across the main diagonal."""
    return np.transpose(image, axes=(1,0,2))

def flip_anti_diagonal(image):
    """Transposes tensor across the anti-diagonal."""
    return np.rot90(np.transpose(image, axes=(1,0,2)), k=1)

def augment_image_full_8x(image):
    """
    Generates D8 dihedral group transformations (8 variants).
    Outputs: Original, Rot90/180/270, flipud, fliplr, diag, anti-diag
    """
    out = []
    out.append(image)
    out.append(np.rot90(image,k=1))
    out.append(np.rot90(image,k=2))
    out.append(np.rot90(image,k=3))
    out.append(np.flipud(image))
    out.append(np.fliplr(image))
    out.append(flip_main_diagonal(image))
    out.append(flip_anti_diagonal(image))
    return np.array(out)


def partial_augmentation(X, y, augment_flags, num_classes, subfolders_array=None):
    """
    Master augmentation router evaluating FLIP_AUG_MODE to dispatch to the 
    appropriate tensor expansion logic.
    """
    mode = FLIP_AUG_MODE.lower()
    if mode == "none":
        X_out = np.array(X)
        y_out = to_categorical(y, num_classes=num_classes)
        if subfolders_array is not None:
            return X_out, y_out, y, np.array(subfolders_array)
        else:
            return X_out, y_out, y, None
    elif mode == "4x":
        return partial_augmentation_4x(X, y, augment_flags, num_classes, subfolders_array)
    else:
        return partial_augmentation_8x(X, y, augment_flags, num_classes, subfolders_array)


def partial_augmentation_4x(X, y, flags, num_classes, subfolders_array=None):
    """Applies orthogonal expansion exclusively to flagged image arrays."""
    out_imgs = []
    out_lbls = []
    out_subf = [] if subfolders_array is not None else None

    for i,(img,lbl,do_aug) in enumerate(zip(X,y,flags)):
        if do_aug:
            variants = augment_image_flips_4x(img)
            out_imgs.extend(variants)
            out_lbls.extend([lbl]*len(variants))
            if out_subf is not None:
                out_subf.extend([subfolders_array[i]]*len(variants))
        else:
            out_imgs.append(img)
            out_lbls.append(lbl)
            if out_subf is not None:
                out_subf.append(subfolders_array[i])

    out_imgs = np.array(out_imgs)
    out_lbls = np.array(out_lbls)
    out_lbls_cat = to_categorical(out_lbls, num_classes=num_classes)

    if out_subf is not None:
        out_subf = np.array(out_subf)
        return out_imgs, out_lbls_cat, out_lbls, out_subf
    else:
        return out_imgs, out_lbls_cat, out_lbls, None


def partial_augmentation_8x(X, y, flags, num_classes, subfolders_array=None):
    """Applies full 8-way expansion exclusively to flagged image arrays."""
    out_imgs = []
    out_lbls = []
    out_subf = [] if subfolders_array is not None else None

    for i,(img,lbl,do_aug) in enumerate(zip(X,y,flags)):
        if do_aug:
            variants = augment_image_full_8x(img)
            out_imgs.extend(variants)
            out_lbls.extend([lbl]*len(variants))
            if out_subf is not None:
                out_subf.extend([subfolders_array[i]]*len(variants))
        else:
            out_imgs.append(img)
            out_lbls.append(lbl)
            if out_subf is not None:
                out_subf.append(subfolders_array[i])

    out_imgs = np.array(out_imgs)
    out_lbls = np.array(out_lbls)
    out_lbls_cat = to_categorical(out_lbls, num_classes=num_classes)

    if out_subf is not None:
        out_subf = np.array(out_subf)
        return out_imgs, out_lbls_cat, out_lbls, out_subf
    else:
        return out_imgs, out_lbls_cat, out_lbls, None


def random_augmentation_for_one_image(img):
    """Fallback generator applying a single random transformation matrix."""
    mode = FLIP_AUG_MODE.lower()
    if mode=="none":
        return img
    elif mode=="4x":
        arr = augment_image_flips_4x(img)
    else:
        arr = augment_image_full_8x(img)

    rnd_idx = np.random.randint(0,len(arr))
    return arr[rnd_idx]

def random_augmentation_generator(X, y, flags, batch_size, num_classes):
    """Memory-efficient fallback generator for OOM environments."""
    num_samples = len(X)
    idxs = np.arange(num_samples)

    while True:
        np.random.shuffle(idxs)
        for start in range(0,num_samples,batch_size):
            end = min(start+batch_size, num_samples)
            b_idxs = idxs[start:end]

            Xb = []
            yb = []
            for i in b_idxs:
                if flags[i]:
                    Xb.append(random_augmentation_for_one_image(X[i]))
                else:
                    Xb.append(X[i])
                yb.append(y[i])

            Xb = np.array(Xb)
            yb = to_categorical(yb, num_classes=num_classes)
            yield Xb,yb


# Dataset loading and label mapping

def check_if_should_augment(top_level_class, subfolders):
    """Validates specific sub-folders against AUGMENTATION_RULES dict."""
    if AUGMENTATION_RULES=='all':
        return True
    if top_level_class not in AUGMENTATION_RULES:
        return False
    rule = AUGMENTATION_RULES[top_level_class]
    if rule is None:
        return True
    elif len(subfolders)==0:
        return False
    else:
        first_sub = subfolders[0]
        return (first_sub in rule)

def load_images_with_flags(base_dir, img_size=(60,60), color_mode='grayscale'):
    """
    Crawls raw image directory, applying numeric labels based on folder structure
    and toggling binary classification consolidation if flagged.
    """
    possible_classes = [
        f for f in sorted(os.listdir(base_dir))
        if os.path.isdir(os.path.join(base_dir,f))
    ]

    if BINARY_CLASSIFICATION:
        folder_to_label = {}
        for f in possible_classes:
            if f == "no_dislocation":
                folder_to_label[f] = 0
            else:
                folder_to_label[f] = 1
        class_names = ["no_dislocation", "dislocation"]
        print("[INFO] Binary classification mode => 0:'no_dislocation', 1:'dislocation'")
    else:
        folder_to_label = {}
        for lbl_idx, f in enumerate(possible_classes):
            folder_to_label[f] = lbl_idx
        class_names = possible_classes

    all_imgs = []
    all_lbls = []
    all_subf = []
    flags    = []

    print(f"\nLoading from: {base_dir}")
    for folder in possible_classes:
        path = os.path.join(base_dir, folder)
        label_for_this_folder = folder_to_label[folder]

        for root,_,files in os.walk(path):
            relp = os.path.relpath(root, path)
            if relp==".":
                subfolders=[]
            else:
                subfolders=relp.split(os.sep)

            do_aug = check_if_should_augment(folder, subfolders)
            subf_name = subfolders[0] if len(subfolders)>0 else ''

            for file in files:
                if file.lower().endswith('.tiff') or file.lower().endswith('.tif'):
                    fpath = os.path.join(root,file)
                    img = load_img(fpath, target_size=img_size, color_mode=color_mode)
                    arr = img_to_array(img)
                    all_imgs.append(arr)
                    all_lbls.append(label_for_this_folder)
                    all_subf.append(subf_name)
                    flags.append(do_aug)

        if not BINARY_CLASSIFICATION:
            print(f"Class '{folder}' => label={label_for_this_folder}")

    all_imgs = np.array(all_imgs)
    all_lbls = np.array(all_lbls)
    flags    = np.array(flags,dtype=bool)
    all_subf = np.array(all_subf)

    print(f"Total images loaded={len(all_imgs)}")

    return all_imgs, all_lbls, flags, all_subf, class_names


def print_distribution_by_class_and_subfolder(labels_array, subfolders_array, class_names):
    """Outputs terminal diagnostic logs mapping class imbalance after augmentation."""
    if len(labels_array.shape)>1 and labels_array.shape[1]==len(class_names):
        labels_array = np.argmax(labels_array,axis=1)

    from collections import defaultdict
    dct = defaultdict(lambda: defaultdict(int))
    for lbl,subf in zip(labels_array, subfolders_array):
        dct[lbl][subf]+=1

    print("\n--- Post-Aug Distribution by (Class,Subfolder) ---")
    for lbl in sorted(dct.keys()):
        cname = class_names[lbl]
        sub_map = dct[lbl]
        print(f"\nClass '{cname}' (label={lbl}):")
        tot=0
        for sf in sorted(sub_map.keys()):
            cnt = sub_map[sf]
            print(f"   Subfolder='{sf}': {cnt}")
            tot+=cnt
        print(f"   --> total in '{cname}': {tot}")
    print("---------------------------------------------------\n")

def display_sample_augmentations(X, flags, num_examples=3):
    """Generates preview plots to visually verify tensor transformation mappings."""
    import matplotlib.pyplot as plt
    mode = FLIP_AUG_MODE.lower()
    if mode=="none":
        print("[INFO] No augmentation => no samples.")
        return
    aug_idxs = np.where(flags==True)[0]
    if len(aug_idxs)==0:
        print("[INFO] No images flagged for augmentation.")
        return
    chosen = np.random.choice(aug_idxs, size=min(num_examples,len(aug_idxs)), replace=False)
    for cidx in chosen:
        if mode=="4x":
            arr = augment_image_flips_4x(X[cidx])
            titles=["Original","FlipLR","FlipUD","FlipBoth"]
        else:
            arr = augment_image_full_8x(X[cidx])
            titles=["Original","Rot90","Rot180","Rot270","FlipUD","FlipLR","FlipDiag","FlipAntiDiag"]
        for i,v in enumerate(arr):
            plt.figure()
            plt.imshow(v.reshape(IMG_SIZE[0], IMG_SIZE[1]), cmap='gray')
            plt.title(f"idx={cidx}, {titles[i]}")
            plt.axis('off')
            plt.show()

def display_batches_with_confidence(X_val, y_pred_classes, y_true_classes, y_pred, class_names, batch_size=25):
    """Renders validation images flagged green/red based on correct label alignment."""
    if DISPLAY_IMAGES:
        tot = len(X_val)
        nb = tot//batch_size + int(tot%batch_size>0)
        for b in range(nb):
            st = b*batch_size
            en = min(st+batch_size, tot)
            for i in range(st,en):
                plt.figure()
                plt.imshow(X_val[i].reshape(IMG_SIZE[0], IMG_SIZE[1]), cmap='gray')
                color = 'green' if y_pred_classes[i]==y_true_classes[i] else 'red'
                conf  = np.max(y_pred[i])*100
                if y_pred_classes[i]<len(class_names):
                    pred_lbl = class_names[y_pred_classes[i]]
                else:
                    pred_lbl = "unknown"
                true_lbl = class_names[y_true_classes[i]]
                plt.title(f"Pred: {pred_lbl} ({conf:.1f}%)\nTrue: {true_lbl}",
                          fontsize=8, color=color)
                plt.axis('off')
                plt.show()
    else:
        print("[INFO] DISPLAY_IMAGES=False => skipping.")


# Model architectures and checkpoint callbacks

class CustomModelCheckpoint(tf.keras.callbacks.Callback):
    """Custom callback routing intermediate save states with dynamic suffix naming."""
    def __init__(self, epochs_to_save, save_dir, model_name_prefix, date_str):
        super().__init__()
        self.epochs_to_save = epochs_to_save
        self.save_dir = save_dir
        self.model_name_prefix = model_name_prefix
        self.date_str = date_str
        self.aug_suffix = get_augmentation_suffix()

    def on_epoch_end(self, epoch, logs=None):
        ep = epoch+1
        if ep in self.epochs_to_save:
            if USE_OPENMAX:
                fname = f"{self.model_name_prefix}_{self.date_str}_epoch_{ep}_{self.aug_suffix}_openmax.h5"
            else:
                fname = f"{self.model_name_prefix}_{self.date_str}_epoch_{ep}_{self.aug_suffix}.h5"

            fpath = os.path.join(self.save_dir, fname)
            self.model.save(fpath)
            print(f"\n[Checkpoint] Saved model at epoch {ep} -> {fpath}")

def build_baseline_cnn(num_classes):
    """Lightweight 3-layer CNN optimized for structural artifact detection."""
    model = Sequential([
        Conv2D(32,(3,3),activation='relu', input_shape=(IMG_SIZE[0],IMG_SIZE[1],1)),
        MaxPooling2D((2,2)),
        Conv2D(64,(3,3),activation='relu'),
        MaxPooling2D((2,2)),
        Conv2D(128,(3,3),activation='relu'),
        MaxPooling2D((2,2)),
        Flatten(),
        Dense(128,activation='relu'),
        Dropout(0.5),
        Dense(64,activation='relu'),
        Dense(num_classes,activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def build_deeper_cnn(num_classes):
    """4-layer CNN augmented with batch normalization layers for unstable variance."""
    model = Sequential([
        Conv2D(32,(3,3),activation='relu', input_shape=(IMG_SIZE[0],IMG_SIZE[1],1)),
        BatchNormalization(),
        MaxPooling2D((2,2)),
        Conv2D(64,(3,3),activation='relu'),
        BatchNormalization(),
        MaxPooling2D((2,2)),
        Conv2D(128,(3,3),activation='relu'),
        BatchNormalization(),
        MaxPooling2D((2,2)),
        Conv2D(256,(3,3),activation='relu'),
        BatchNormalization(),
        MaxPooling2D((2,2)),
        Flatten(),
        Dense(256,activation='relu'),
        Dropout(0.5),
        Dense(128,activation='relu'),
        Dropout(0.5),
        Dense(num_classes,activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def build_custom_resnet_gray(num_classes):
    inp = Input(shape=(IMG_SIZE[0],IMG_SIZE[1],1))
    x = Conv2D(32,(3,3),activation='relu',padding='same')(inp)
    x = BatchNormalization()(x)
    x = Flatten()(x)
    x = Dense(256,activation='relu')(x)
    x = Dropout(0.5)(x)
    out= Dense(num_classes,activation='softmax')(x)
    mod= Model(inp,out)
    mod.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return mod

def build_custom_inception_gray(num_classes):
    inp = Input(shape=(IMG_SIZE[0],IMG_SIZE[1],1))
    x = Conv2D(32,(3,3),activation='relu',padding='same')(inp)
    x = BatchNormalization()(x)
    x = Flatten()(x)
    x = Dense(256,activation='relu')(x)
    x = Dropout(0.5)(x)
    out= Dense(num_classes,activation='softmax')(x)
    mod= Model(inp,out)
    mod.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return mod


# OpenMax / unknown class detection

def build_feature_extractor(model):
    """Isolates the penultimate Dense layer (activation vector) for EVT calibration."""
    layer_for_embedding = model.layers[-2].output
    return Model(inputs=model.input, outputs=layer_for_embedding)

def gather_training_embeddings(feature_extractor, X_train, y_train):
    """Routes validation array through the feature extractor to capture Mean Activation Vectors."""
    if len(y_train.shape)>1 and y_train.shape[1]>1:
        y_idx = np.argmax(y_train, axis=1)
    else:
        y_idx = y_train
    emb = feature_extractor.predict(X_train,batch_size=32)
    dct = defaultdict(list)
    for e,c in zip(emb,y_idx):
        dct[c].append(e)
    return dct

def fit_weibull_per_class(embeddings_dict):
    """
    Fits a Weibull continuous probability distribution curve to the distance metric 
    variance between known classes, defining the threshold for the 'Unknown' margin.
    """
    class_mavs = {}
    weibull_params = {}
    for c, arr_list in embeddings_dict.items():
        arr = np.stack(arr_list, axis=0)
        mav = arr.mean(axis=0)
        class_mavs[c]=mav

        dists = np.linalg.norm(arr - mav, axis=1)
        shape, loc, scale = weibull_min.fit(dists, floc=0)
        weibull_params[c] = (shape,scale)
    return class_mavs, weibull_params

def openmax_inference_single(x, model, feat_ext, class_mavs, weibull_params, alpha=5):
    """
    Reweights standard SoftMax categorical output using the Weibull CDF to insert 
    an N+1 'Unknown' tensor probability slice.
    """
    raw_logits = model.predict(x[np.newaxis,...], verbose=0)[0]
    emb       = feat_ext.predict(x[np.newaxis,...], verbose=0)[0]
    sorted_idx= np.argsort(raw_logits)[::-1]
    top_alpha = sorted_idx[:alpha]

    adj_logits = np.copy(raw_logits)
    unknown_logit = 0.0

    for c_idx in top_alpha:
        shape,scale = weibull_params[c_idx]
        mav = class_mavs[c_idx]
        dist= np.linalg.norm(emb-mav)
        cdf_val = weibull_min.cdf(dist, shape,0,scale)

        old_val = adj_logits[c_idx]
        new_val = old_val*cdf_val
        diff    = old_val-new_val

        unknown_logit += diff
        adj_logits[c_idx] = new_val

    fin_logits = np.append(adj_logits, unknown_logit)
    exps = np.exp(fin_logits - np.max(fin_logits))
    probs= exps/np.sum(exps)
    return probs

def openmax_inference_batch(X, model, feat_ext, class_mavs, weibull_params, alpha=5):
    """Vectorizes openmax_inference_single across evaluation batches."""
    out = []
    for i in range(len(X)):
        p = openmax_inference_single(X[i], model, feat_ext, class_mavs, weibull_params, alpha=alpha)
        out.append(p)
    return np.array(out)


# Training / fallback orchestration

def train_and_evaluate_model(build_fn,
                             model_name_prefix,
                             X_train_,
                             y_train_,
                             X_val_,
                             y_val_,
                             max_epochs,
                             batch_size,
                             epochs_to_save,
                             class_names,
                             final_save_dir,
                             date_str):
    """
    Standard model training loop. Generates accuracy logs and executes OpenMax 
    re-calibration callbacks if OpenMax toggles are active.
    """
    aug_suffix = get_augmentation_suffix()
    num_classes = len(class_names)

    print("\n=================================================")
    print(f"Training Model: {model_name_prefix}")
    print(f"Binary Mode={BINARY_CLASSIFICATION} => #classes={num_classes}")
    print(f"Aug Mode: {FLIP_AUG_MODE} ({aug_suffix}), USE_OPENMAX={USE_OPENMAX}")
    print(f"Max epochs={max_epochs}, batch_size={batch_size}")
    print(f"Saving checkpoints at {epochs_to_save}")
    print("=================================================\n")

    model = build_fn(num_classes=num_classes)

    chkpt_cb = CustomModelCheckpoint(
        epochs_to_save=epochs_to_save,
        save_dir=final_save_dir,
        model_name_prefix=model_name_prefix,
        date_str=date_str
    )

    st_time = time.time()
    history = model.fit(
        X_train_,
        y_train_,
        validation_data=(X_val_, y_val_),
        epochs=max_epochs,
        batch_size=batch_size,
        callbacks=[chkpt_cb],
        verbose=2
    )
    train_time = time.time()-st_time

    if USE_OPENMAX:
        final_name = f"{model_name_prefix}_{date_str}_epoch_{max_epochs}_{aug_suffix}_openmax.h5"
    else:
        final_name = f"{model_name_prefix}_{date_str}_epoch_{max_epochs}_{aug_suffix}.h5"

    final_path = os.path.join(final_save_dir, final_name)
    model.save(final_path)
    print(f"\n[Final Save] => {final_path}")
    print(f"Training took {train_time:.2f}s\n")

    if not USE_OPENMAX:
        val_loss,val_acc = model.evaluate(X_val_, y_val_, verbose=0)
        print(f"[{model_name_prefix}] Val acc={val_acc:.4f}")

        y_pred = model.predict(X_val_)
        y_pc = np.argmax(y_pred,axis=1)
        y_tc = np.argmax(y_val_,axis=1)
        display_batches_with_confidence(X_val_, y_pc, y_tc, y_pred, class_names,25)

        return history, model
    else:
        print("[OpenMax] Fitting Weibull on train embeddings...\n")
        feat_ext = build_feature_extractor(model)
        train_emb_dict = gather_training_embeddings(feat_ext, X_train_, y_train_)
        class_mavs, weibull_params = fit_weibull_per_class(train_emb_dict)

        print("[OpenMax] Re-calibrating val...\n")
        openmax_probs = openmax_inference_batch(X_val_, model, feat_ext, class_mavs, weibull_params, alpha=5)
        y_pred_om = np.argmax(openmax_probs, axis=1)
        y_true_om = np.argmax(y_val_, axis=1)

        unknown_mask = (y_pred_om == len(class_names))
        n_unk = np.sum(unknown_mask)
        print(f"[OpenMax] # 'unknown' predicted = {n_unk}/{len(X_val_)}")

        valid_idx = np.where(~unknown_mask)[0]
        y_pred_known = y_pred_om[valid_idx]
        y_true_known = y_true_om[valid_idx]
        correct_known= np.sum(y_pred_known==y_true_known)
        if len(valid_idx)>0:
            acc_known = correct_known/len(valid_idx)
            print(f"[OpenMax] Acc among knowns={acc_known:.4f}")
        else:
            print("[OpenMax] All predicted unknown => can't measure known accuracy")

        display_batches_with_confidence(X_val_, y_pred_om, y_true_om, openmax_probs, class_names,25)
        return history, model

def train_model_safely(model_name_prefix,
                       build_fn,
                       max_epochs,
                       epochs_to_save,
                       class_names,
                       final_save_dir,
                       date_str):
    """
    Execution wrapper that attempts full dataset memory ingestion but gracefully 
    falls back to a slow, batch-level randomized generator if an OOM 
    (ResourceExhaustedError) fault occurs during tensor mapping.
    """
    batch_size = 32
    num_classes = len(class_names)
    print(f"[INFO] Attempt partial augmentation => {FLIP_AUG_MODE}")

    try:
        X_aug, y_aug_cat, y_aug_indices, subf_aug = partial_augmentation(
            X_train_orig,
            y_train_orig,
            train_flags,
            num_classes=num_classes,
            subfolders_array=train_subfolders
        )
        X_val_ = X_val_orig
        y_val_cat = to_categorical(y_val_orig, num_classes=num_classes)

        if subf_aug is not None:
            print_distribution_by_class_and_subfolder(y_aug_indices, subf_aug, class_names)

        hist, model_obj = train_and_evaluate_model(
            build_fn=build_fn,
            model_name_prefix=model_name_prefix,
            X_train_=X_aug,
            y_train_=y_aug_cat,
            X_val_=X_val_,
            y_val_=y_val_cat,
            max_epochs=max_epochs,
            batch_size=batch_size,
            epochs_to_save=epochs_to_save,
            class_names=class_names,
            final_save_dir=final_save_dir,
            date_str=date_str
        )

    except (tf.errors.ResourceExhaustedError, MemoryError) as e:
        print(f"[OOM] => fallback random generator.")
        tf.keras.backend.clear_session()
        gc.collect()

        model = build_fn(num_classes=num_classes)

        train_gen = random_augmentation_generator(
            X_train_orig,
            y_train_orig,
            train_flags,
            batch_size,
            num_classes=num_classes
        )
        steps_per_epoch = max(len(X_train_orig)//batch_size,1)
        val_data = (X_val_orig, to_categorical(y_val_orig, num_classes=num_classes))

        chkpt_cb = CustomModelCheckpoint(
            epochs_to_save=epochs_to_save,
            save_dir=final_save_dir,
            model_name_prefix=model_name_prefix,
            date_str=date_str
        )

        st_time = time.time()
        hist = model.fit(
            train_gen,
            epochs=max_epochs,
            steps_per_epoch=steps_per_epoch,
            validation_data=val_data,
            callbacks=[chkpt_cb],
            verbose=2
        )
        el_time = time.time()-st_time

        aug_suffix = get_augmentation_suffix()
        if USE_OPENMAX:
            final_name = f"{model_name_prefix}_{date_str}_epoch_{max_epochs}_{aug_suffix}_openmax.h5"
        else:
            final_name = f"{model_name_prefix}_{date_str}_epoch_{max_epochs}_{aug_suffix}.h5"

        final_path = os.path.join(final_save_dir, final_name)
        model.save(final_path)
        print(f"[Fallback] => {final_path}")
        print(f"Train time={el_time:.2f}s\n")

        if not USE_OPENMAX:
            val_loss,val_acc = model.evaluate(val_data[0], val_data[1], verbose=0)
            print(f"[Fallback] Val acc={val_acc:.4f}")
            y_pred = model.predict(val_data[0])
            y_pc   = np.argmax(y_pred,axis=1)
            y_tc   = np.argmax(val_data[1],axis=1)
            display_batches_with_confidence(val_data[0], y_pc, y_tc, y_pred, class_names,25)
            model_obj = model
        else:
            print("[OpenMax-Fallback] Fitting Weibull...")
            feat_ext = build_feature_extractor(model)
            train_emb_dict = gather_training_embeddings(feat_ext, X_train_orig, to_categorical(y_train_orig, num_classes=num_classes))
            class_mavs, weibull_params = fit_weibull_per_class(train_emb_dict)

            print("[OpenMax-Fallback] Inference on val.")
            openmax_probs = openmax_inference_batch(val_data[0], model, feat_ext, class_mavs, weibull_params, alpha=5)
            y_pred_om = np.argmax(openmax_probs,axis=1)
            y_true_om = np.argmax(val_data[1],axis=1)
            unk_mask  = (y_pred_om==num_classes)
            n_unk     = np.sum(unk_mask)
            print(f"[OpenMax-Fallback] # unknown={n_unk}")
            valid_idx = np.where(~unk_mask)[0]
            if len(valid_idx)>0:
                ckn = np.sum(y_pred_om[valid_idx]==y_true_om[valid_idx])
                print(f"[OpenMax-Fallback] Acc known={ckn/len(valid_idx):.4f}")
            display_batches_with_confidence(val_data[0], y_pred_om, y_true_om, openmax_probs, class_names,25)
            model_obj = model

    return hist, model_obj


# Main execution pipeline

all_images, all_labels, augment_flags, all_subfolders, CLASS_NAMES = load_images_with_flags(
    base_dir=BASE_DIR,
    img_size=IMG_SIZE,
    color_mode='grayscale'
)

NUM_CLASSES = len(CLASS_NAMES)

X_train_orig, X_val_orig, y_train_orig, y_val_orig, train_flags, val_flags, train_subfolders, val_subfolders = \
    train_test_split(
        all_images,
        all_labels,
        augment_flags,
        all_subfolders,
        test_size=VAL_SPLIT,
        random_state=42
    )

print(f"\n[INFO] train size={len(X_train_orig)}, val size={len(X_val_orig)}")

u_tr, c_tr = np.unique(y_train_orig, return_counts=True)
print("[Train Dist - pre aug]:")
for l,c in zip(u_tr,c_tr):
    print(f"  {CLASS_NAMES[l]} => {c}")

u_val, c_val = np.unique(y_val_orig, return_counts=True)
print("[Val Dist]:")
for l,c in zip(u_val,c_val):
    print(f"  {CLASS_NAMES[l]} => {c}")

if SHOW_AUG_SAMPLES:
    display_sample_augmentations(X_train_orig, train_flags, num_examples=3)

MODEL_BUILDERS = {
    "baseline_cnn": build_baseline_cnn,
    "deeper_cnn":   build_deeper_cnn,
    "custom_resnet_gray": build_custom_resnet_gray,
    "custom_inception_gray": build_custom_inception_gray
}

all_histories={}

for model_name in TRAIN_MODEL_TYPES:
    if model_name not in MODEL_BUILDERS:
        print(f"[WARN] No builder for '{model_name}', skipping.")
        continue

    if USE_OPENMAX:
        out_dir = f"{model_name}_{NUM_CLASSES}_class_openmax"
    else:
        out_dir = f"{model_name}_{NUM_CLASSES}_class"

    final_save_dir = os.path.join(ROOT_SAVE_DIR, out_dir, current_date)
    os.makedirs(final_save_dir, exist_ok=True)

    print(f"\n[INFO] => {model_name}, saving to:\n   {final_save_dir}\n")

    build_fn = MODEL_BUILDERS[model_name]

    hist, model_obj = train_model_safely(
        model_name_prefix=model_name,
        build_fn=build_fn,
        max_epochs=GLOBAL_MAX_EPOCHS,
        epochs_to_save=CHECKPOINT_EPOCHS,
        class_names=CLASS_NAMES,
        final_save_dir=final_save_dir,
        date_str=current_date
    )

    all_histories[model_name] = hist
    del model_obj
    del hist
    tf.keras.backend.clear_session()
    gc.collect()

print("\n[INFO] All training done!")
print("Models saved under =>", ROOT_SAVE_DIR)