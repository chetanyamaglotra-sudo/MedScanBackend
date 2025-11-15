from tensorflow.keras.applications import VGG16
from tensorflow.keras import layers, models, optimizers
from config import LEARNING_RATE, FINE_TUNE_AT

def build_vgg_classifier(input_shape, num_classes, fine_tune_at=FINE_TUNE_AT):
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=input_shape)
    
    # Freeze all layers first
    for layer in base_model.layers:
        layer.trainable = False

    # Unfreeze the last few layers for fine-tuning
    for layer in base_model.layers[-fine_tune_at:]:
        layer.trainable = True

    x = layers.Flatten()(base_model.output)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    output = layers.Dense(num_classes, activation='softmax')(x)

    model = models.Model(inputs=base_model.input, outputs=output)
    model.compile(
        optimizer=optimizers.Adam(learning_rate=LEARNING_RATE),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model
