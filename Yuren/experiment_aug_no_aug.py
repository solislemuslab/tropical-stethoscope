'''
This is the script to randomly picked a given number of sonotypes
with sample sizes above a given number.
Then, augment the samples and classification.
Record the classification performance 
between augmentation and no augmentation.

As the script is to be run on the server with parallel computing,
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
import csv
import tensorflow as tf


"""Normalization"""


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
            (returned number of samples will be 1 + 15*aug_num)
    @param: augment_range is the threshold used for augmentations, default to 0.1
    @return: augment_specs_func is the list of augmented spectrograms
    @return: augment_aux_func is the list of  auxiliary input corresponds to the spectrograms
    @return: augment_sono_func is the list of sonotypes input corresponds to the spectrograms
    '''
    # augment_range = 0.1
    augment_specs_func = []
    augment_aux_func = []
    augment_sono_func = []

    for i in range(len(specs)):
        # generate random non-repeated index array for augmentation,
        # in 5% to 10% of the size of the original spectrogram
        # 224 * 224 is the image size
        indices = np.arange(int(224 * augment_range / 3 * 2),
                            int(224 * augment_range))
        np.random.shuffle(indices)
        indices = indices[:aug_num]

        # augment each spec and add to list
        cur_spec = np.copy(specs[i])
        # add itself to the list
        if (len(augment_specs_func)):
            augment_specs_func = np.append(
                augment_specs_func, [cur_spec], axis=0)
        else:
            augment_specs_func.append(cur_spec)
        # augment_specs_func.append(cur_spec)

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
                [aux_input[i]], 1 + 15 * aug_num, axis=0), axis=0)
        else:
            augment_aux_func = np.repeat(
                [aux_input[i]], 1 + 15 * aug_num, axis=0)

        augment_sono_func = np.append(augment_sono_func, np.repeat(
            sonotypes[i], 1 + 15 * aug_num), axis=0)

    return augment_specs_func, augment_aux_func, augment_sono_func


"""Methods for classification"""


def get_samples(numUsed, num_pick=49):
    '''
    Get samples, separate the trainign, test, and validation
    return the separated sets for training, validatioon, and testing

    @param: numUsed, the number of sonotypes to use
    @param: num_pick, the number of sample size to use
    @return: typeUsed, the sonotypes ramdomly picked
    @return: specs, the training spectrograms
    @return: aux_input, the training auxiliary inputs
    @return: sonotypes, the training sonotypes formated in number from 0 - (numUsed-1)
    @return: x_test, the input for testing
    @return: x_val, the input for validation
    @return: cat_y_test, the catgorical output for test
    @return: cat_y_val, the catgorical output for validation
    '''
    # get groups
    typeUsed = []
    # while len(typeUsed) < numUsed:
    index = 0
    while len(typeUsed) < 6:  # always use top six for birs so far
        cur_type = s_unique[s_freq_order][index]
        if sono2group[cur_type] == b'b':  # use bird for now
            typeUsed.append(cur_type)
        index += 1

    random.shuffle(typeUsed)
    typeUsed = typeUsed[:numUsed]

    # if do not specify types
    # type_index = np.argwhere((s_freq_desc >= num_pick) & (
    #     s_unique[s_freq_order] > 0)).flatten()
    # random.shuffle(type_index)
    # type_index = np.sort(type_index[:numUsed])
    # typeUsed = s_unique[s_freq_order][type_index]
    # print("type index:", type_index)

    print("type used: ", typeUsed)

    specs = []
    aux_input = []
    sonotypes = []
    spec_test = []
    aux_test = []
    y_test = []
    spec_val = []
    aux_val = []
    y_val = []

    for i in range(len(typeUsed)):
        # get index of the current type of spec
        cur_index = np.argwhere(sonotypes_h5 == typeUsed[i]).flatten()
        random.shuffle(cur_index)

        # get index of the current type of spec
        # 80% training, 10% validation, 10% testing
        cur_index = np.argwhere(sonotypes_h5 == typeUsed[i]).flatten()
        random.shuffle(cur_index)
        cur_index_resized = cur_index[:int(num_pick * 0.8)]
        test_index = cur_index[int(num_pick * 0.8): int(num_pick * 0.9)]
        val_index = cur_index[int(num_pick * 0.9): num_pick]

        if len(specs):
            # not null
            specs = np.append(specs, specs_h5[cur_index_resized], axis=0)
            aux_input = np.append(
                aux_input, aux_input_h5[cur_index_resized], axis=0)
            sonotypes = np.append(sonotypes, np.repeat(i, int(num_pick * 0.8)))
            spec_test = np.append(spec_test, specs_h5[test_index], axis=0)
            aux_test = np.append(aux_test, aux_input_h5[test_index], axis=0)
            spec_val = np.append(spec_val, specs_h5[val_index], axis=0)
            aux_val = np.append(aux_val, aux_input_h5[val_index], axis=0)
            y_test = np.append(y_test, np.repeat(i, len(test_index)))
            y_val = np.append(y_val, np.repeat(i, len(val_index)))
        else:
            specs = specs_h5[cur_index_resized]
            aux_input = aux_input_h5[cur_index_resized]
            sonotypes = np.repeat(i, int(num_pick * 0.8))
            spec_test = specs_h5[test_index]
            aux_test = aux_input_h5[test_index]
            spec_val = specs_h5[val_index]
            aux_val = aux_input_h5[val_index]
            y_test = np.repeat(i, len(test_index))
            y_val = np.repeat(i, len(val_index))

    # formulate the test and validation set
    x_test = [spec_test, aux_test]
    x_val = [spec_val, aux_val]
    cat_y_test = to_categorical(pd.factorize(
        y_test)[0], num_classes=len(typeUsed))
    cat_y_val = to_categorical(pd.factorize(
        y_val)[0], num_classes=len(typeUsed))

    print("train, test, val size:", specs.shape[0], len(
        cat_y_test), len(cat_y_val))

    return typeUsed, specs, aux_input, sonotypes, x_test, x_val, cat_y_test, cat_y_val


class TestCallback(keras.callbacks.Callback):
    '''
    The class used to stop the training process early if the
    validation loss is 0
    '''

    def __init__(self, test_data):
        self.test_data = test_data

    def on_epoch_end(self, epoch, logs={}):
        x, y = self.test_data
        loss, acc = self.model.evaluate(x, y, verbose=0)
        print('\nTesting loss: {}, acc: {}\n'.format(loss, acc))


def gen(specs, aux_input, sonotypes):
    '''
    generator functiion for fit_generator
    augment the samples and yield the augmented samples to train the model

    @param: specs, the list of spectrograms
    @param: aux_input, the list of auxiliary input corresponding to the spectrograms
    @param: sonotypes, the list of sonotypes corresponding to the spectrograms
    '''
    # augment_specs, augment_aux, augment_sono =  augment(specs_seperated[i], aux_seperated[i], sono_seperated[i], 1)
    while 1:
        # shuffle data
        indices = np.arange(len(sonotypes))
        np.random.shuffle(indices)
        # # use 30 of all data per epoch
        # indices = indices[:30]
        step_len = 4

        for i in range(len(specs) // step_len):
            step_min = i * step_len
            step_max = min((i + 1) * step_len, len(specs))

            augment_specs, augment_aux, augment_sono = augment(
                specs[indices][step_min: step_max], aux_input[indices][step_min: step_max], sonotypes[indices][step_min: step_max], 1)
            augment_specs_normal = normalize(augment_specs)
            cat_y_train = to_categorical(
                augment_sono, num_classes=len(typeUsed))

            yield {'input_1': np.array([augment_specs_normal])[0], 'aux_input': np.array([augment_aux])[0]}, np.array([cat_y_train])[0]


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

    # create the dictionary for sonotypes and groups
    sono2group = dict(zip(sonotypes_h5, groups_h5))

    # get the data for top k sonotypes
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
    all_result = []
    all_result_no = []
    all_result_aug = []
    all_type_used = []
    for i in range(20):  # repeat time for each run
        for j in range(2, 6):  # sonotype number
            print("\niteration: ", i, ", cur #classes: ", j)
            tf.keras.backend.clear_session()
            typeUsed, specs, aux_input, sonotypes, x_test, x_val, cat_y_test, cat_y_val = None, None, None, None, None, None, None, None

            typeUsed, specs, aux_input, sonotypes, x_test, x_val, cat_y_test, cat_y_val = get_samples(
                j)
            all_type_used.append(typeUsed)
            print("all type used", all_type_used)

            # train w/o augmentation
            model = None
            keras.backend.clear_session()
            model = VGG19(weights='imagenet', include_top=False,
                          input_shape=(224, 224, 3))
            model = build_finetune_model(model,
                                         [config["dropout"], config["dropout"]],
                                         [config["hidden"], config["hidden"]],
                                         len(typeUsed))

            filepath = 'model_group.hdf5'
            # stop after 15 epoches without reduction of validation loss
            earlystop = EarlyStopping(
                monitor='val_loss', mode='min', verbose=1, patience=15)
            # save the model with lowest validation loss
            checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1,
                                         save_best_only=True, save_weights_only=False, mode='auto', period=1)
            opt = Adam(lr=config["learn_rate"])
            model.compile(optimizer=opt, loss='categorical_crossentropy',
                          metrics=['accuracy'])
            cat_y_train = to_categorical(sonotypes, num_classes=len(typeUsed))

            history = model.fit(x=[specs, aux_input], y=cat_y_train, validation_data=(
                x_val, cat_y_val), epochs=300, verbose=2, callbacks=[checkpoint, earlystop, TestCallback((x_test, cat_y_test))])

            # test: load the model with lowest validation loss
            model = None
            keras.backend.clear_session()
            model = load_model(filepath)

            results = model.evaluate(x=x_test, y=cat_y_test)
            all_result_no.append("%s, %s\n" % ("; ".join(map(str, typeUsed.flatten())), ", ".join(map(str, results))))
            # all_result_no.append(results)
            print("test loss, test acc:", results)
            print(all_result_no)

            # with augmentation
            model = None
            keras.backend.clear_session()
            model = VGG19(weights='imagenet', include_top=False,
                          input_shape=(224, 224, 3))
            model = build_finetune_model(model,
                                         [config["dropout"], config["dropout"]],
                                         [config["hidden"], config["hidden"]],
                                         len(typeUsed))

            filepath = 'model_group.hdf5'
            # stop after 15 epoches without reduction of validation loss
            earlystop = EarlyStopping(
                monitor='val_loss', mode='min', verbose=1, patience=15)
            # save the model with lowest validation loss
            checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1,
                                         save_best_only=True, save_weights_only=False, mode='auto', period=1)
            opt = Adam(lr=config["learn_rate"])
            model.compile(optimizer=opt, loss='categorical_crossentropy',
                          metrics=['accuracy'])

            history = model.fit_generator(gen(specs, aux_input, sonotypes),
                                          steps_per_epoch=len(specs) // 4, epochs=300, validation_data=(x_val, cat_y_val), verbose=2, callbacks=[checkpoint, earlystop, TestCallback((x_test, cat_y_test))])

            # test: load the model with lowest validation loss
            model = None
            keras.backend.clear_session()
            model = load_model(filepath)

            results = model.evaluate(x=x_test, y=cat_y_test)
            
            # all_result_aug.append(results)
            all_result_aug.append("%s, %s\n" % ("; ".join(map(str, typeUsed.flatten())), ", ".join(map(str, results))))
            print("test loss, test acc:", results)
            
            # print here in case of unexpected stop of the server
            print(all_result_aug)
            print(all_result_no)
