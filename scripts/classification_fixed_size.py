'''
This is the script randomly picked a given number of sonotypes
with a fixed sample sizes.
Then, augment the samples to 250 per sonotype, train,
and check accuracy.

As the script is to be run on the server with parallel computing on condor,
we print the output instead of saving it to specific file

Last updated: 05/26/2021 by Yuren Sun
'''


# %tensorflow_version 1.x
import os
import h5py
import random
import numpy as np
import pandas as pd
import keras
from keras.optimizers import Adam, SGD
from keras.applications import VGG19
from keras.utils import to_categorical
from keras.models import Model, load_model
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Dense, Activation, Flatten, Dropout, Input, concatenate
import cv2
from sklearn.metrics import roc_auc_score
import tensorflow as tf


def normalize(specs):
    '''
    Linear normalization of the data
    @param: specs is the list of spcetrograms to normalize
    @return: the normalized spectrograms
    '''
    return_specs = []
    for i in range(len(specs)):
        # make a copy to ensure not changing the original spectrogram
        cur_spec = np.copy(specs[i])
        s_min = np.amin(cur_spec)
        s_max = np.amax(cur_spec)
        return_specs.append((cur_spec - s_min)/(s_max - s_min) * 255)

    return return_specs


# Augmention functions
# for each function, make a copy of the original spectrogram
# to ensure that we do not change the original one
# return all the augmented spectrograms in lists for consistency


def time_chop(spec, rand_start):
    '''
    chop the spectrogram on x axis (time) from the right
    @param: spec, the spectrogram to chop
    @param: rand_start: the randomed index to start chopping
    @return: the list of augmented spectrograms
    '''
    time_chopped_spec = np.copy(spec)
    time_chopped_spec[:, 224 - rand_start:, :] = 0

    return [time_chopped_spec]


def freq_chop(spec, rand_start):
    '''
    chop the spectrogram on y axis (frequency) from the top
    @param: spec, the spectrogram to chop
    @param: rand_start: the randomed index to start chopping
    @return: the list of augmented spectrograms
    '''

    freq_chopped_spec = np.copy(spec)
    freq_chopped_spec[0:rand_start, :, :] = 0

    return [freq_chopped_spec]


def four_chop(spec, rand_start):
    '''
    chop the spectrogram on four sides
    @param: spec, the spectrogram to chop
    @param: rand_start: the randomed index to start chopping
    @return: the list of augmented spectrograms
    '''
    four_chopped_spec = np.copy(spec)
    four_chopped_spec[0: rand_start, :, :] = 0  # top
    four_chopped_spec[:, 224 - rand_start:, :] = 0  # right
    four_chopped_spec[224 - rand_start:, :, :] = 0  # bottom
    four_chopped_spec[:, 0: rand_start, :] = 0  # left

    return [four_chopped_spec]


def add_noises(spec):
    '''
    add noise to the spectrogram with 1/3 ratio
    @param: spec, the spectrogram to chop
    @return: the list of augmented spectrograms
    '''
    # add noise from light rian -2, rain -3, heavy rain -4, thunder -5, aircraft -6, chainsaw -7, and car/truck -8
    return_specs = []
    noise_sonos = [-2, -3, -4, -5, -6, -7, -8]

    for i in range(len(noise_sonos)):
        noises_index = np.argwhere(sonotypes_h5 == noise_sonos[i]).flatten()
        noises = specs_h5[noises_index]
        # randomly pick a noise sample
        index = random.randint(0, len(noises) - 1)
        # normalize sound and noise, add them together with 1/3 ratio
        noise = normalize(np.array(noises[index]) / 3)
        return_specs.append(np.add(normalize([np.copy(spec)])[0], noise))

    return return_specs


def translate(spec, roll_start):
    '''
    roll the spectrogram up and down
    @param: spec, the spectrogram to chop
    @param: roll_start, the index to start rolling
    @return: the list of augmented spectrograms
    '''
    return_specs = []
    return_specs.append(np.roll(np.copy(spec), -roll_start, axis=0))
    return_specs.append(np.roll(np.copy(spec), roll_start, axis=0))

    return return_specs


def widen(spec, widen_index):
    '''
    widen the spectrogram
    @param: spec, the spectrogram to chop
    @param: widen_index, the index to decide the start and end of
            the spectrogram to widen
    @return: the list of augmented spectrograms
    '''
    return_specs = []
    widen_time_spec = cv2.resize(np.copy(spec).astype(
        'float32'), (224 + widen_index, 224))
    widen_freq_spec = cv2.resize(np.copy(spec).astype(
        'float32'), (224, 224 + widen_index))

    return_specs.append(
        widen_time_spec[:, widen_index // 2: -widen_index // 2, :])
    return_specs.append(
        widen_freq_spec[widen_index // 2: -widen_index // 2, :, :])

    return return_specs


def squeeze(spec, squeeze_index):
    '''
    squeeze the spectrogram
    @param: spec, the spectrogram to chop
    @param: widen_index, the index to decide the start and end of
            the spectrogram to widen
    @return: the list of augmented spectrograms
    '''
    squeezed = cv2.resize(np.copy(spec).astype('float32'),
                          (224 - squeeze_index, 224 - squeeze_index))
    squeeze_spec = np.zeros([224, 224, 3])
    squeeze_spec[squeeze_index//2: - squeeze_index // 2,
                 squeeze_index//2: - squeeze_index // 2, :] = squeezed

    return [squeeze_spec]


def augment(specs, aux_input, sonotypes, aug_num, augment_range=0.1):
    '''
    call all the augment methods on the spectrograms

    @param: specs is the list of spectrograms to augment from
    @param: aux_input is the list of auxiliary input corresponds to the spectrograms
    @param: sonotypes is the list of sonnotypes corresponds to the spectrograms
    @param: aug_num is the number of sets of augmented spectrograms
            (returned number of samples will be 15*aug_num)
    @param: augment_range is the threshold used for augmentations, default to 0.1
    @return: augment_specs_func is the list of augmented spectrograms
    @return: augment_aux_func is the list of  auxiliary input corresponds to the spectrograms
    @return: augment_sono_func is the list of sonotypes input corresponds to the spectrograms
    '''

    # lists to return
    augment_specs_func = []
    augment_aux_func = []
    augment_sono_func = []

    # augment each spec and add to list
    for i in range(len(specs)):
        # generate random non-repeated index array for augmentation,
        # in 5% to 10% of the size of the original spectrogram
        # 224 * 224 is the image size
        indices = np.arange(int(224 * augment_range / 2),
                            int(224 * augment_range))
        np.random.shuffle(indices)
        indices = indices[:aug_num]

        cur_spec = np.copy(specs[i])
        # # add itself to the list, not added in this file
        # if (len(augment_specs_func)):
        #     augment_specs_func = np.append(
        #         augment_specs_func, [cur_spec], axis=0)
        # else:
        #     augment_specs_func.append(cur_spec)
        # # augment_specs_func.append(cur_spec)

        for index in indices:
            # chop
            augment_specs_func = np.append(
                augment_specs_func, time_chop(np.copy(cur_spec), index), axis=0)
            augment_specs_func = np.append(
                augment_specs_func, freq_chop(np.copy(cur_spec), index), axis=0)
            augment_specs_func = np.append(
                augment_specs_func, four_chop(np.copy(cur_spec), index), axis=0)

            # widen + squeeze
            augment_specs_func = np.append(
                augment_specs_func, squeeze(np.copy(cur_spec), index), axis=0)
            augment_specs_func = np.append(
                augment_specs_func, widen(np.copy(cur_spec), index), axis=0)

            # noise
            augment_specs_func = np.append(
                augment_specs_func, add_noises(np.copy(cur_spec)), axis=0)

            # translate
            augment_specs_func = np.append(
                augment_specs_func, translate(np.copy(cur_spec), index), axis=0)

        # total 1 + 15 * aug_num augmented, repeat the sono and aux
        if (len(augment_aux_func)):
            augment_aux_func = np.append(augment_aux_func, np.repeat(
                [aux_input[i]], 15 * aug_num, axis=0), axis=0)
        else:
            augment_aux_func = np.repeat(
                [aux_input[i]], 15 * aug_num, axis=0)

        augment_sono_func = np.append(augment_sono_func, np.repeat(
            sonotypes[i], 15 * aug_num), axis=0)

    return augment_specs_func, augment_aux_func, augment_sono_func


def get_samples(numUsed, min_type, fix_size):
    '''
    Get samples, augment to 250 samples per sonotypes
    return the separated sets for training, validatioon, and testing

    @param: numUsed, the number of sonotypes to use
    @param: min_type, the number of minimum_type
    @param: fix_size True if use the same sample size for all sonotypes.
                False otherwise.
    @return: typeUsed, the sonotypes ramdomly picked
    @return: sizes, the list of original sample sizes for the sonotypes
    @return: specs, the training spectrograms
    @return: aux_input, the training auxiliary inputs
    @return: sonotypes, the training sonotypes formated in number from 0 - (numUsed-1)
    @return: x_test, the input for testing
    @return: x_val, the input for validation
    @return: cat_y_test, the catgorical output for test
    @return: cat_y_val, the catgorical output for validation
    '''
    # get groups
    aug_goal = 250
    # randomly pick sonotype with sample size > min_type
    # and are not noises (sonotype > 0)
    type_index = np.argwhere((s_freq_desc >= min_type) & (
        s_unique[s_freq_order] > 0)).flatten()
    random.shuffle(type_index)
    type_index = np.sort(type_index[:numUsed])
    typeUsed = s_unique[s_freq_order][type_index]

    print("type index:", type_index)
    print("type used: ", typeUsed)

    # aug
    specs = []
    aux_input = []
    sonotypes = []
    spec_test = []
    aux_test = []
    y_test = []
    spec_val = []
    aux_val = []
    y_val = []
    sizes = []

    # no aug
    specs_no = []
    aux_input_no = []
    sonotypes_no = []
    spec_test_no = []
    aux_test_no = []
    y_test_no = []
    spec_val_no = []
    aux_val_no = []
    y_val_no = []


    for i in range(len(typeUsed)):
        # get index of the current type of spec
        cur_index = np.argwhere(sonotypes_h5 == typeUsed[i]).flatten()
        random.shuffle(cur_index)

        if fix_size:
            cur_index = cur_index[:min_type]
        sizes.append(len(cur_index))  # append the used size

        # decide the text and val size,
        # split in train: validation: test = 80%:10%:10%
        # at least 1 for both test and val
        text_val_size = max(1, int(len(cur_index) * 0.1))
        cur_index_resized = cur_index[:len(cur_index) - 2 * text_val_size]
        test_index = cur_index[len(cur_index) - 2 *
                               text_val_size: len(cur_index) - text_val_size]
        val_index = cur_index[len(cur_index) - text_val_size: len(cur_index)]
        print("sonotype, len of cur:", typeUsed[i], len(cur_index))
        print("train, test, val size: ", len(cur_index_resized),
              len(test_index), len(val_index))

        # augment to aug goal separatedly for training, validation, and testing sets
        # train， 80% of aug goal, each augmentation pass augment the sample to another 15 sampls
        augment_num = (int(aug_goal * 0.8) // len(cur_index_resized)) // 15 + 1
        # train
        augment_specs, augment_aux, augment_sono = augment(
            specs_h5[cur_index_resized], aux_input_h5[cur_index_resized], np.repeat(i, len(cur_index_resized)), augment_num)

        # first in augmented set is itself
        aug_index = np.arange(2, len(augment_specs))
        random.shuffle(aug_index)
        aug_index = aug_index[:int(aug_goal * 0.8) - len(cur_index_resized)]
        print("test aug size", len(aug_index))

        if len(specs):
            specs = np.concatenate(
                (specs, specs_h5[cur_index_resized], augment_specs[aug_index]), axis=0)
            aux_input = np.concatenate(
                (aux_input, aux_input_h5[cur_index_resized], augment_aux[aug_index]), axis=0)
            sonotypes = np.append(sonotypes, np.repeat(
                i, len(cur_index_resized) + len(aug_index)))

            specs_no = np.concatenate(
                (specs_no, specs_h5[cur_index_resized]), axis=0)
            aux_input_no = np.concatenate(
                (aux_input_no, aux_input_h5[cur_index_resized]), axis=0)
            sonotypes_no = np.append(sonotypes_no, np.repeat(
                i, len(cur_index_resized)))
        else:
            specs = np.concatenate(
                (specs_h5[cur_index_resized], augment_specs[aug_index]), axis=0)
            aux_input = np.concatenate(
                (aux_input_h5[cur_index_resized], augment_aux[aug_index]), axis=0)
            sonotypes = np.repeat(i, len(cur_index_resized) + len(aug_index))

            specs_no = np.copy(specs_h5[cur_index_resized])
            aux_input_no = np.copy(aux_input_h5[cur_index_resized])
            sonotypes_no = np.repeat(i, len(cur_index_resized))

        # test and val
        augment_num = (int(aug_goal * 0.1) // text_val_size) // 15 + 1
        # test
        augment_specs, augment_aux, augment_sono = None, None, None
        augment_specs, augment_aux, augment_sono = augment(
            specs_h5[test_index], aux_input_h5[test_index], np.repeat(i, len(test_index)), augment_num)
        aug_index = np.arange(2, len(augment_specs))
        random.shuffle(aug_index)
        aug_index = aug_index[:int(aug_goal * 0.1) - text_val_size]
        print("test aug size", len(aug_index))

        if len(spec_test):
            spec_test = np.concatenate(
                (spec_test, specs_h5[test_index], augment_specs[aug_index]), axis=0)
            aux_test = np.concatenate(
                (aux_test, aux_input_h5[test_index], augment_aux[aug_index]), axis=0)
            y_test = np.append(y_test, np.repeat(
                i, len(test_index) + len(aug_index)))
            spec_test_no = np.concatenate(
                (spec_test_no, specs_h5[test_index]), axis=0)
            aux_test_no = np.concatenate(
                (aux_test_no, aux_input_h5[test_index]), axis=0)
            y_test_no = np.append(y_test_no, np.repeat(i, len(test_index)))

        else:
            spec_test = np.concatenate(
                (specs_h5[test_index], augment_specs[aug_index]), axis=0)
            aux_test = np.concatenate(
                (aux_input_h5[test_index], augment_aux[aug_index]), axis=0)
            y_test = np.repeat(i, len(test_index) + len(aug_index))
            spec_test_no = np.copy(specs_h5[test_index])
            aux_test_no = np.copy(aux_input_h5[test_index])
            y_test_no = np.repeat(i, len(test_index))

        # val
        augment_specs, augment_aux, augment_sono = None, None, None
        augment_specs, augment_aux, augment_sono = augment(
            specs_h5[val_index], aux_input_h5[val_index], np.repeat(i, len(val_index)), augment_num)
        aug_index = np.arange(2, len(augment_specs))
        random.shuffle(aug_index)
        aug_index = aug_index[:int(aug_goal * 0.1) - text_val_size]

        if len(spec_val):
            spec_val = np.concatenate(
                (spec_val, specs_h5[val_index], augment_specs[aug_index]), axis=0)
            aux_val = np.concatenate(
                (aux_val, aux_input_h5[val_index], augment_aux[aug_index]), axis=0)
            y_val = np.append(y_val, np.repeat(
                i, len(val_index) + len(aug_index)))

            spec_val_no = np.concatenate(
                (spec_val_no, specs_h5[val_index]), axis=0)
            aux_val_no = np.concatenate(
                (aux_val_no, aux_input_h5[val_index]), axis=0)
            y_val_no = np.append(y_val_no, np.repeat(i, len(val_index)))
        else:
            spec_val = np.concatenate(
                (specs_h5[val_index], augment_specs[aug_index]), axis=0)
            aux_val = np.concatenate(
                (aux_input_h5[val_index], augment_aux[aug_index]), axis=0)
            y_val = np.repeat(i, len(val_index) + len(aug_index))

            spec_val_no = np.copy(specs_h5[val_index])
            aux_val_no = np.copy(aux_input_h5[val_index])
            y_val_no = np.repeat(i, len(val_index))

        print(specs.shape, len(aux_input), len(sonotypes), len(spec_test), len(
            aux_test), len(y_test), len(spec_val), len(aux_val), len(y_val))

    x_test = [spec_test, aux_test]
    x_val = [spec_val, aux_val]
    cat_y_test = to_categorical(pd.factorize(
        y_test)[0], num_classes=len(typeUsed))
    cat_y_val = to_categorical(pd.factorize(
        y_val)[0], num_classes=len(typeUsed))

    print("train, test, val size:", specs.shape[0], len(
        cat_y_test), len(cat_y_val))

    return typeUsed, sizes, specs, aux_input, sonotypes, x_test, x_val, cat_y_test, cat_y_val,\
        specs_no, aux_input_no, sonotypes_no, x_test_no, x_val_no, cat_y_test_no, cat_y_val_no, y_test_no, y_test


class TestCallback(keras.callbacks.Callback):
    '''
    The class used to test and check test stats
    during the training process
    '''

    def __init__(self, test_data):
        self.test_data = test_data

    def on_epoch_end(self, epoch, logs={}):
        x, y = self.test_data
        loss, acc = self.model.evaluate(x, y, verbose=0)
        print('\nTesting loss: {}, acc: {}\n'.format(loss, acc))


def build_finetune_model(base_model, dropouts, fc_layers, num_classes):
    '''
    finetune the model, freeze teh top layers,
    add dropouts, dense layers,
    another input layer for auxiliary input
    and concatenate it with the flatten layer
    '''
    # fix the base layers
    for layer in base_model.layers:
        layer.trainable = False

    # add flatten layer
    x = base_model.output
    x = Flatten()(x)

    # add input layer for auxiliary frequency and time
    auxiliary_input = Input(shape=(4,), name='aux_input')
    x = concatenate([x, auxiliary_input])

    # add dense and dropout layer at last
    for fc, drop in zip(fc_layers, dropouts):
        x = Dense(fc, activation='relu')(x)
        x = Dropout(drop)(x)

    # final dense layer for output
    predictions = Dense(num_classes, activation='softmax')(x)

    finetune_model = Model(
        inputs=[base_model.input, auxiliary_input], outputs=predictions)

    return finetune_model


if __name__ == "__main__":
    # Read the dataset
    f = h5py.File('whole_data_1110.hdf5', "r")
    specs_h5 = np.array(f["specs"]).astype("float32")
    sonotypes_h5 = np.array(f["sonotypes"]).astype("float32")
    times_h5 = np.array(f["times"]).astype("float32")
    freqs_h5 = np.array(f["freqs"]).astype("float32")
    groups_h5 = np.array(f["groups"])
    selection_h5 = np.array(f["selections"])
    f.close()

    # append x_times an x_freqs to be auxiliary_input
    aux_input_h5 = np.append(times_h5, freqs_h5, axis=1)

    # # create the dictionary for sonotypes and groups, not used here
    # sono2group = dict(zip(sonotypes_h5, groups_h5))

    # unique sonotypes and sample sizes
    s_unique, s_freq = np.unique(sonotypes_h5, return_counts=True)
    s_freq_order = np.argsort(s_freq)[::-1]
    s_freq_desc = s_freq[s_freq_order]

    config = dict(
        dropout=0.5,
        hidden=1024,
        learn_rate=0.00001,
        epochs=30,
    )

    numUsed = 6
    fixed = False
    all_result = []
    all_result_no = []
    all_result_aug = []
    all_type_used = []

    # i for the fixed size for the sonotypes
    for i in range(3, 20):
        print("\niteration: ", i)
        # clear last session
        tf.keras.backend.clear_session()
        typeUsed, sizes, specs, aux_input, sonotypes, x_test, x_val, cat_y_test, cat_y_val = None, None, None, None, None, None, None, None, None

        typeUsed, sizes, specs, aux_input, sonotypes, x_test, x_val, cat_y_test, cat_y_val,specs_no,aux_input_no, sonotypes_no, x_test_no, x_val_no, cat_y_test_no, cat_y_val_no, y_test_no, y_test = get_samples(
            numUsed, i, fixed)

        all_type_used.append(typeUsed)
        # print("all type used", all_type_used)

        # Augmented
        # load model
        model = None
        keras.backend.clear_session()
        model = VGG19(weights='imagenet', include_top=False,
                      input_shape=(224, 224, 3))
        model = build_finetune_model(model,
                                     [config["dropout"], config["dropout"]],
                                     [config["hidden"], config["hidden"]],
                                     len(typeUsed))

        # earlystopping, checkpoiint, before
        filepath = 'model_group.hdf5'
        earlystop = EarlyStopping(
            monitor='val_loss', mode='min', verbose=1, patience=15)
        checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1,
                                     save_best_only=True, save_weights_only=False, mode='auto', period=1)
        opt = Adam(lr=config["learn_rate"])
        model.compile(optimizer=opt, loss='categorical_crossentropy',
                      metrics=['accuracy'])
        cat_y_train = to_categorical(sonotypes, num_classes=len(typeUsed))

        # training
        history = model.fit(x=[specs, aux_input], y=cat_y_train, validation_data=(
            x_val, cat_y_val), epochs=300, verbose=0, callbacks=[checkpoint, earlystop, TestCallback((x_test, cat_y_test))])

        # test: load the model with best weights
        model = None
        keras.backend.clear_session()
        model = load_model(filepath)

        results = model.evaluate(x=x_test, y=cat_y_test)
        # for roc/auc
        y_score = model.predict(x_test)
        y_class = [typeUsed[i]
                   for i in np.argmax(y_score, axis=1)]  # predicted class
        y_true = [typeUsed[i] for i in y_test]  # true class
        roc_score = roc_auc_score(
            y_test, y_score, sample_weight=None, multi_class="ovo")

        all_result.append(results)
        # print("test loss, test acc:", results)
        print("all result used: ", all_result)
        print("all type used: ", all_type_used)

        # typeuse, sizes, results(acc,loss),roc, y_score, y_class,y_true
        all_result_aug.append("%s, %s, %s, %s, %s, %s, %s\n" % ("; ".join(map(str, typeUsed.flatten())),
                                                      "; ".join(map(str, sizes)),
                                                      ", ".join(map(str, results)),
                                                      str(roc_score),
                                                      ";".join(" ".join(str(num) for num in sub) for sub in y_score),
                                                      ";".join([str(x) for x in y_class]),
                                                      ";".join([str(x) for x in y_true])
                                                      ))
        print("".join(all_result_aug))

        # no aug
        # load model
        model = None
        keras.backend.clear_session()
        model = VGG19(weights='imagenet', include_top=False,
                      input_shape=(224, 224, 3))
        model = build_finetune_model(model,
                                     [config["dropout"], config["dropout"]],
                                     [config["hidden"], config["hidden"]],
                                     len(typeUsed))

        # earlystopping, checkpoiint, before
        filepath = 'model_group_no.hdf5'
        earlystop = EarlyStopping(
            monitor='val_loss', mode='min', verbose=1, patience=15)
        checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1,
                                     save_best_only=True, save_weights_only=False, mode='auto', period=1)
        opt = Adam(lr=config["learn_rate"])
        model.compile(optimizer=opt, loss='categorical_crossentropy',
                      metrics=['accuracy'])
        cat_y_train_no = to_categorical(sonotypes_no, num_classes=len(typeUsed))

        # training
        history = model.fit(x=[specs_no, aux_input_no], y=cat_y_train_no, validation_data=(
            x_val_no, cat_y_val_no), epochs=300, verbose=0, callbacks=[checkpoint, earlystop, TestCallback((x_test_no, cat_y_test_no))])

        # test: load the model with best weights
        model = None
        keras.backend.clear_session()
        model = load_model(filepath)

        results = model.evaluate(x=x_test_no, y=cat_y_test_no)
        # for roc/auc
        y_score = model.predict(x_test_no)
        y_class = [typeUsed[i]
                   for i in np.argmax(y_score, axis=1)]  # predicted class
        y_true = [typeUsed[i] for i in y_test_no]  # true class
        roc_score = roc_auc_score(
            y_test_no, y_score, sample_weight=None, multi_class="ovo")

        # typeuse, sizes, results(acc,loss),roc, y_score, y_class,y_true
        all_result_no.append("%s, %s, %s, %s, %s, %s, %s\n" % ("; ".join(map(str, typeUsed.flatten())),
                                                      "; ".join(map(str, sizes)),
                                                      ", ".join(map(str, results)),
                                                      str(roc_score),
                                                      ";".join(" ".join(str(num) for num in sub) for sub in y_score),
                                                      ";".join([str(x) for x in y_class]),
                                                      ";".join([str(x) for x in y_true])
                                                      ))
        print("".join(all_result_no))

    print("Fixed size or not:", fixed)
    print("aug")
    print("".join(all_result_aug))

    print("no aug")
    print("".join(all_result_no))
