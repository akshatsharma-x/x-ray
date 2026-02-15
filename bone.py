import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.applications import MobileNetV2
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import cv2
import os
from tensorflow.keras.preprocessing import image
from google.colab import files

train_dir = '/content/drive/My Drive/Bone/train/'
validation_dir = '/content/drive/My Drive/Bone/val/'

img_width, img_height = 224, 224
batch_size = 32

train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    rotation_range=10,
    horizontal_flip=True)

val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary',
    shuffle=True)

validation_generator = val_datagen.flow_from_directory(
    validation_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary',
    shuffle=False)

class_indices = train_generator.class_indices
class_indices

train_generator.filenames[:20]

base_model = MobileNetV2(weights='imagenet',
                         include_top=False,
                         input_shape=(224,224,3))

base_model.trainable = False

inputs = base_model.input
x = base_model.output

x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(128, activation='relu')(x)
x = layers.Dropout(0.4)(x)
outputs = layers.Dense(1, activation='sigmoid')(x)

model = tf.keras.Model(inputs, outputs)
model.summary()



model.compile(optimizer=optimizers.Adam(1e-4),
              loss='binary_crossentropy',
              metrics=['accuracy'])

history = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=20)


base_model.trainable = True

fine_tune_at = 100

for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False

for layer in base_model.layers[fine_tune_at:]:
    layer.trainable = True
model.compile(
    optimizer=optimizers.Adam(1e-5),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

print("Fine-tuning starting from layer:", fine_tune_at)

history_ft = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=2
)


loss, acc = model.evaluate(validation_generator)
print("Validation Accuracy:", acc)
print("Validation Loss:", loss)

validation_generator.reset()
y_pred = (model.predict(validation_generator) > 0.5).astype("int32")
y_true = validation_generator.classes

print(classification_report(y_true, y_pred))


cm = confusion_matrix(y_true, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()


def find_last_conv(model):
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer.name
        if hasattr(layer, 'layers'):
            for sub in reversed(layer.layers):
                if isinstance(sub, tf.keras.layers.Conv2D):
                    return sub.name
    raise ValueError("No Conv2D layer found")

last_conv = find_last_conv(model)
print("Last Conv Layer:", last_conv)


import tensorflow as tf
import numpy as np
import cv2
from tensorflow.keras.models import Model

def find_last_conv_layer_obj(model):
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer
        if hasattr(layer, "layers"):
            for sub in reversed(layer.layers):
                if isinstance(sub, tf.keras.layers.Conv2D):
                    return sub
    raise ValueError("No Conv2D layer found in model or its sublayers.")

def find_real_input_tensor(model):
    try:
        base = globals().get("base_model", None)
        if base is not None and hasattr(base, "input"):
            return base.input
    except Exception:
        pass

    try:
        return model.input
    except Exception:
        pass

    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.InputLayer):
            return layer.output
    raise ValueError("Could not determine a valid input tensor for the model.")

def build_grad_model_for_gradcam(model):
    conv_layer_obj = find_last_conv_layer_obj(model)
    conv_layer_name = conv_layer_obj.name
    print("Detected last Conv2D layer object:", conv_layer_name)

    conv_output_tensor = conv_layer_obj.output

    real_input = find_real_input_tensor(model)
    print("Using input tensor:", real_input)

    grad_model = Model(inputs=real_input, outputs=[conv_output_tensor, model.output])
    return grad_model, conv_layer_name

grad_model, detected_conv_name = build_grad_model_for_gradcam(model)


def compute_gradcam_heatmap(img_array, grad_model, upsample_to=(224,224)):
    with tf.GradientTape() as tape:
        conv_outputs, preds = grad_model(img_array)
        score = preds[:, 0]
    grads = tape.gradient(score, conv_outputs)
    if grads is None:
        raise RuntimeError("Gradients returned None — graph disconnected.")

    pooled_grads = tf.reduce_mean(grads, axis=(0,1,2))
    conv_outputs = conv_outputs[0]

    heatmap = tf.zeros(conv_outputs.shape[0:2])
    for i in range(conv_outputs.shape[-1]):
        heatmap += pooled_grads[i] * conv_outputs[:,:,i]

    heatmap = tf.maximum(heatmap, 0)
    heatmap /= (tf.reduce_max(heatmap) + 1e-10)
    heatmap = heatmap.numpy()
    heatmap = cv2.resize(heatmap, upsample_to)
    return np.clip(heatmap, 0, 1)


def overlay_heatmap_on_image(img, heatmap, alpha=0.4):
    heatmap_uint8 = np.uint8(255 * heatmap)
    colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    img_uint8 = np.uint8(255 * img)
    overlay = cv2.addWeighted(colored, alpha, img_uint8, 1-alpha, 0)
    overlay = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
    return overlay


imgs, lbls = next(validation_generator)

img = imgs[0]
true_label = lbls[0]

x = np.expand_dims(img, axis=0)

heatmap = compute_gradcam_heatmap(x, grad_model, (img.shape[1], img.shape[0]))
overlay = overlay_heatmap_on_image(img, heatmap)

plt.figure(figsize=(12,5))
plt.subplot(1,3,1); plt.imshow(img); plt.title("Input Image"); plt.axis('off')
plt.subplot(1,3,2); plt.imshow(heatmap, cmap='jet'); plt.title("Grad-CAM Heatmap"); plt.axis('off')
plt.subplot(1,3,3); plt.imshow(overlay); plt.title("Overlay"); plt.axis('off')
plt.show()




def predict_and_gradcam(img, grad_model, model):
    x = np.expand_dims(img, axis=0)
    pred_prob = model.predict(x)[0][0]
    pred_label = "FRACTURED" if pred_prob > 0.5 else "NORMAL"

    heatmap = compute_gradcam_heatmap(x, grad_model, (img.shape[1], img.shape[0]))
    overlay = overlay_heatmap_on_image(img, heatmap)

    plt.figure(figsize=(14,5))
    plt.subplot(1,3,1); plt.imshow(img); plt.title("Input X-ray"); plt.axis('off')
    plt.subplot(1,3,2); plt.imshow(heatmap, cmap='jet'); plt.title("Grad-CAM Heatmap"); plt.axis('off')
    plt.subplot(1,3,3); plt.imshow(overlay);
    plt.title(f"Overlay\nPrediction: {pred_label}\nConfidence: {pred_prob:.3f}")
    plt.axis('off')
    plt.show()




imgs, lbls = next(validation_generator)
predict_and_gradcam(imgs[0], grad_model, model)


def full_diagnostic_panel(img, model, grad_model):
    x = np.expand_dims(img, axis=0)
    prob = float(model.predict(x)[0][0])
    pred_label = "FRACTURED" if prob > 0.5 else "NORMAL"

    heatmap = compute_gradcam_heatmap(x, grad_model, (img.shape[1], img.shape[0]))
    overlay = overlay_heatmap_on_image(img, heatmap)

    fig, axes = plt.subplots(2, 2, figsize=(12,10))

    axes[0,0].imshow(img)
    axes[0,0].set_title("Original X-ray")
    axes[0,0].axis('off')

    axes[0,1].imshow(heatmap, cmap='inferno')
    axes[0,1].set_title("Grad-CAM Heatmap")
    axes[0,1].axis('off')

    axes[1,0].imshow(overlay)
    axes[1,0].set_title("Heatmap Overlay")
    axes[1,0].axis('off')

    axes[1,1].bar(["Normal", "Fractured"], [1-prob, prob], color=["green", "red"])
    axes[1,1].set_ylim([0,1])
    axes[1,1].set_title("Prediction Probabilities")
    axes[1,1].set_ylabel("Probability")

    plt.suptitle(f"Prediction: {pred_label} ({prob:.3f})", fontsize=16)
    plt.tight_layout()
    plt.show()



full_diagnostic_panel(imgs[0], model, grad_model)



def gradcam_grid(images, grad_model, model, rows=2, cols=4):
    plt.figure(figsize=(16,8))
    for i in range(rows*cols):
        img = images[i]
        x = np.expand_dims(img, axis=0)
        heatmap = compute_gradcam_heatmap(x, grad_model, (img.shape[1], img.shape[0]))
        overlay = overlay_heatmap_on_image(img, heatmap)

        plt.subplot(rows, cols, i+1)
        plt.imshow(overlay)
        plt.axis('off')
        plt.title(f"Image {i+1}")

    plt.suptitle("Grad-CAM Overlay Grid", fontsize=18)
    plt.tight_layout()
    plt.show()


imgs, lbls = next(validation_generator)
gradcam_grid(imgs, grad_model, model)


import os

def save_gradcam_batch(images, output_folder="gradcam_outputs"):
    os.makedirs(output_folder, exist_ok=True)

    for idx, img in enumerate(images):
        x = np.expand_dims(img, axis=0)
        heatmap = compute_gradcam_heatmap(x, grad_model, (img.shape[1], img.shape[0]))
        overlay = overlay_heatmap_on_image(img, heatmap)

        cv2.imwrite(f"{output_folder}/gradcam_{idx}.jpg", cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

    print(f"Saved {len(images)} Grad-CAM images to {output_folder}/")



save_gradcam_batch(imgs)

def gradcam_filmstrip(img, model, grad_model):
    x = np.expand_dims(img, axis=0)
    heatmap = compute_gradcam_heatmap(x, grad_model, (img.shape[1], img.shape[0]))
    overlay = overlay_heatmap_on_image(img, heatmap)

    fig = plt.figure(figsize=(12,4))

    titles = ["Input", "Grad-CAM", "Overlay"]
    imgs_plot = [img, heatmap, overlay]

    for i in range(3):
        ax = fig.add_subplot(1,3,i+1)
        ax.imshow(imgs_plot[i], cmap=None if i!=1 else 'jet')
        ax.axis("off")
        ax.set_title(titles[i])

    plt.tight_layout()
    plt.show()



gradcam_filmstrip(imgs[0], model, grad_model)


def prediction_bar_with_gradcam(img, model, grad_model):
    x = np.expand_dims(img, axis=0)
    prob = model.predict(x)[0][0]
    pred_label = "Fractured" if prob > 0.5 else "Normal"

    heatmap = compute_gradcam_heatmap(x, grad_model, (img.shape[1], img.shape[0]))
    overlay = overlay_heatmap_on_image(img, heatmap)

    fig = plt.figure(figsize=(14,6))

    ax1 = fig.add_subplot(1,2,1)
    ax1.imshow(overlay)
    ax1.set_title(f"Prediction: {pred_label}")
    ax1.axis("off")

    ax2 = fig.add_subplot(1,2,2)
    ax2.bar(["Normal", "Fractured"], [1-prob, prob], color=["green", "crimson"])
    ax2.set_ylim(0,1)
    ax2.set_title("Prediction Confidence")
    ax2.set_ylabel("Probability")

    plt.tight_layout()
    plt.show()



def clinical_gradcam(img, model, grad_model):
    x = np.expand_dims(img, axis=0)
    heatmap = compute_gradcam_heatmap(x, grad_model, (img.shape[1], img.shape[0]))
    overlay = overlay_heatmap_on_image(img, heatmap)

    fig, axs = plt.subplots(1, 2, figsize=(12,5), facecolor='black')

    axs[0].imshow(img)
    axs[0].set_title("Original", color='white')
    axs[0].axis('off')

    axs[1].imshow(overlay)
    axs[1].set_title("Grad-CAM Overlay", color='white')
    axs[1].axis('off')

    plt.show()


imgs, lbls = next(validation_generator)

img = imgs[0]

prediction_bar_with_gradcam(img, model, grad_model)

clinical_gradcam(img, model, grad_model)


for i in range(5):
    print(f"---- Image {i} ----")
    prediction_bar_with_gradcam(imgs[i], model, grad_model)


def advanced_probability_bar(img, model, grad_model):
    x = np.expand_dims(img, axis=0)
    prob = float(model.predict(x)[0][0])

    fig, ax = plt.subplots(figsize=(8,4))

    bars = ax.bar(["Normal", "Fractured"], [1-prob, prob],
                  color=["#4CAF50", "#C62828"], edgecolor="black")

    for bar in bars:
        height = bar.get_height()
        ax.annotate(f"{height:.2f}",
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    ha='center', va='bottom', fontsize=12, color='black')

    ax.set_ylim(0,1)
    ax.set_ylabel("Probability")
    ax.set_title("Prediction Confidence (with Value Labels)", fontsize=14)
    plt.show()



advanced_probability_bar(imgs[0], model, grad_model)


def radar_chart_prediction(img, model):
    x = np.expand_dims(img, axis=0)
    prob = float(model.predict(x)[0][0])

    categories = ["Normal", "Fractured"]
    values = [1-prob, prob]

    N = len(categories)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()

    values += values[:1]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(6,6), subplot_kw=dict(polar=True))
    ax.plot(angles, values, linewidth=2)
    ax.fill(angles, values, alpha=0.3)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories)
    ax.set_title("Prediction Radar Chart", fontsize=14)

    plt.show()



radar_chart_prediction(imgs[0], model)


def multi_intensity_gradcam(img, model, grad_model):
    x = np.expand_dims(img, axis=0)
    heatmap = compute_gradcam_heatmap(x, grad_model, (img.shape[1], img.shape[0]))

    fig, axs = plt.subplots(1, 3, figsize=(15,5))
    alphas = [0.3, 0.5, 0.7]

    for i, alpha in enumerate(alphas):
        overlay = overlay_heatmap_on_image(img, heatmap, alpha)
        axs[i].imshow(overlay)
        axs[i].set_title(f"α = {alpha}")
        axs[i].axis('off')

    plt.suptitle("Multi-Intensity Grad-CAM Views", fontsize=16)
    plt.show()



multi_intensity_gradcam(imgs[0], model, grad_model)



def prediction_trend_plot(validation_generator, model, num_samples=30):
    imgs, lbls = next(validation_generator)
    imgs = imgs[:num_samples]

    preds = model.predict(imgs).flatten()

    plt.figure(figsize=(10,4))
    plt.plot(preds, marker='o', linewidth=2)
    plt.ylim(0,1)
    plt.xlabel("Sample Index")
    plt.ylabel("Fracture Probability")
    plt.title("Prediction Trend Across Samples")
    plt.grid(True)
    plt.show()



prediction_trend_plot(validation_generator, model)


def full_advanced_panel(img, model, grad_model):
    x = np.expand_dims(img, axis=0)
    prob = float(model.predict(x)[0][0])

    heatmap = compute_gradcam_heatmap(x, grad_model, (img.shape[1], img.shape[0]))
    overlay = overlay_heatmap_on_image(img, heatmap)

    fig = plt.figure(figsize=(15,8))

    ax1 = fig.add_subplot(2,2,1)
    ax1.imshow(overlay)
    ax1.set_title("Grad-CAM Overlay")
    ax1.axis("off")

    ax2 = fig.add_subplot(2,2,2)
    ax2.imshow(heatmap, cmap="inferno")
    ax2.set_title("Heatmap Only")
    ax2.axis("off")

    ax3 = fig.add_subplot(2,2,3)
    ax3.bar(["Normal", "Fractured"], [1-prob, prob],
            color=["green", "crimson"])
    ax3.set_ylim(0,1)
    ax3.set_title("Prediction Probability")

    categories = ["Normal", "Fractured"]
    angles = np.linspace(0, 2 * np.pi, 2, endpoint=False).tolist()
    rad_vals = [1-prob, prob, 1-prob]
    angles += angles[:1]

    ax4 = fig.add_subplot(2,2,4, polar=True)
    ax4.plot(angles, rad_vals, linewidth=2)
    ax4.fill(angles, rad_vals, alpha=0.3)
    ax4.set_xticks(angles[:-1])
    ax4.set_xticklabels(categories)
    ax4.set_title("Radar Confidence Map")

    plt.tight_layout()
    plt.show()


full_advanced_panel(imgs[4], model, grad_model)

def medical_journal_composite(img, model, grad_model):
    x = np.expand_dims(img, axis=0)
    prob = float(model.predict(x)[0][0])

    heatmap = compute_gradcam_heatmap(x, grad_model, (img.shape[1], img.shape[0]))
    overlay = overlay_heatmap_on_image(img, heatmap)

    fig = plt.figure(figsize=(18,6), facecolor='white')

    ax1 = fig.add_subplot(1,3,1)
    ax1.imshow(img)
    ax1.set_title("A. Original X-ray", fontsize=14)
    ax1.axis('off')

    ax2 = fig.add_subplot(1,3,2)
    ax2.imshow(overlay)
    ax2.set_title("B. Grad-CAM Overlay", fontsize=14)
    ax2.axis('off')

    ax3 = fig.add_subplot(1,3,3)
    ax3.bar(["Normal", "Fractured"], [1-prob, prob], color=["#2E7D32","#B71C1C"])
    ax3.set_ylim(0, 1)
    ax3.set_title("C. Model Confidence", fontsize=14)
    ax3.set_ylabel("Probability")

    plt.tight_layout()
    plt.show()


imgs, lbls = next(validation_generator)

img = imgs[2]

medical_journal_composite(img, model, grad_model)


def radiologist_quad_panel(img, model, grad_model):
    x = np.expand_dims(img, axis=0)
    heatmap = compute_gradcam_heatmap(x, grad_model, (img.shape[1], img.shape[0]))
    overlay = overlay_heatmap_on_image(img, heatmap)

    gray = cv2.cvtColor(np.uint8(img*255), cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 50, 150)

    fig = plt.figure(figsize=(16,8))

    panels = [
        ("A. Original X-ray", img),
        ("B. Grad-CAM Heatmap", heatmap),
        ("C. Edge-enhanced Fracture Map", edges),
        ("D. Overlay Image", overlay)
    ]

    for i,(title,data) in enumerate(panels):
        ax = fig.add_subplot(2,2,i+1)
        cmap = None if i in [0,3] else "jet"
        ax.imshow(data, cmap=cmap)
        ax.set_title(title, fontsize=12)
        ax.axis("off")

    plt.tight_layout()
    plt.show()


radiologist_quad_panel(imgs[0], model, grad_model)

def thermal_gradcam(img, model, grad_model):
    x = np.expand_dims(img, axis=0)
    heatmap = compute_gradcam_heatmap(x, grad_model, (img.shape[1], img.shape[0]))

    heatmap_uint8 = np.uint8(255 * heatmap)
    thermal = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_HOT)
    thermal = cv2.cvtColor(thermal, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(7,7))
    plt.imshow(thermal)
    plt.title("Thermal Medical Grad-CAM", fontsize=14)
    plt.axis("off")
    plt.show()



thermal_gradcam(imgs[0], model, grad_model)


