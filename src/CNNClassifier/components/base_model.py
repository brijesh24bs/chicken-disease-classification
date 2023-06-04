import os
import urllib.request as request
from zipfile import ZipFile
import zipfile
from pathlib import Path
import tensorflow as tf
from CNNClassifier.entity import BaseModelConfig

class BaseModel:
    def __init__(self, config: BaseModelConfig):
        self.config = config

    def get_base_model(self):
        model = tf.keras.applications.vgg16.VGG16(
            input_shape=self.config.params_image_size,
            weights=self.config.params_weights,
            include_top=self.config.params_include_top
            )
        model.save(self.config.base_model_path)

    @staticmethod
    def prepare_full_model(model, classes, freeze_all, freeze_till, learning_rate) -> tf.keras.models.Model:
        if freeze_all:
            for layer in model.layers:
                layer.trainable = False
        elif (freeze_till is not None) and (freeze_till > 0):
            for layer in model.layers[:-freeze_till]:
                layer.trainable = False

        flatten_in = tf.keras.layers.Flatten()(model.output)
        prediction = tf.keras.layers.Dense(
            units=classes,
            activation='softmax'
        )(flatten_in)

        full_model = tf.keras.models.Model(
            inputs=model.input,
            outputs=prediction
        )

        full_model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss=tf.keras.losses.CategoricalCrossentropy(),
            metrics=['accuracy']
        )
        return full_model

    def update_base_model(self):
        model = tf.keras.models.load_model(self.config.base_model_path)
        updated_base_model = self.prepare_full_model(
            model=model,
            classes=self.config.params_classes,
            freeze_all=True,
            freeze_till=None,
            learning_rate=self.config.params_learning_rate
        )
        updated_base_model.save(self.config.updated_base_model_path)

