import streamlit as st
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Set page config
st.set_page_config(page_title="MNIST Optimizer Comparison", layout="wide")

st.title("🧠 MNIST Optimizer Comparison")
st.markdown("""
Compare **Adam**, **RMSprop**, and **SGD** on the MNIST dataset. 
After training, a comparison table of final accuracies will be displayed.
""")

# --- Sidebar Configuration ---
st.sidebar.header("Training Settings")
epochs = st.sidebar.slider("Number of Epochs", 1, 10, 3)
batch_size = st.sidebar.select_slider("Batch Size", options=[64, 128, 256], value=128)
learning_rate = st.sidebar.number_input("Learning Rate", value=0.001, format="%.4f")
# Option to speed up training for demonstration
use_subset = st.sidebar.checkbox("Use data subset (Faster)", value=True)

# --- Data Loading ---
@st.cache_data
def load_data(subset=False):
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    if subset:
        # Use only 10,000 images for speed
        x_train, y_train = x_train[:10000], y_train[:10000]
    
    x_train, x_test = x_train / 255.0, x_test / 255.0
    x_train = x_train[..., tf.newaxis].astype("float32")
    x_test = x_test[..., tf.newaxis].astype("float32")
    return x_train, y_train, x_test, y_test

x_train, y_train, x_test, y_test = load_data(subset=use_subset)

# --- Model Building ---
def create_model():
    return models.Sequential([
        layers.Conv2D(16, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(32, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])

# --- Comparison Logic ---
if st.button('🚀 Run Comparison'):
    optimizers = {
        'Adam': tf.keras.optimizers.Adam(learning_rate=learning_rate),
        'RMSprop': tf.keras.optimizers.RMSprop(learning_rate=learning_rate),
        'SGD': tf.keras.optimizers.SGD(learning_rate=learning_rate)
    }
    
    results = {}
    trained_models = {}
    final_metrics = []
    
    st.write("### ⏳ Training in progress...")
    progress_bar = st.progress(0)
    
    for idx, (name, opt) in enumerate(optimizers.items()):
        model = create_model()
        model.compile(optimizer=opt, 
                      loss='sparse_categorical_crossentropy', 
                      metrics=['accuracy'])
        
        history = model.fit(x_train, y_train, 
                            epochs=epochs, 
                            batch_size=batch_size, 
                            validation_data=(x_test, y_test), 
                            verbose=0)
        
        results[name] = history.history
        trained_models[name] = model
        
        # Capture final accuracy
        final_acc = history.history['accuracy'][-1]
        final_val_acc = history.history['val_accuracy'][-1]
        final_metrics.append({
            "Optimizer": name,
            "Final Train Accuracy": f"{final_acc:.4f}",
            "Final Val Accuracy": f"{final_val_acc:.4f}"
        })
        
        progress_bar.progress((idx + 1) / 3)

    # --- 1. Final Accuracy Summary ---
    st.divider()
    st.header("🏆 Final Accuracy Results")
    st.table(pd.DataFrame(final_metrics))

    # --- 2. Visualization: Accuracy & Loss Graphs ---
    st.header("📊 Performance Metrics")
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    
    for name, hist in results.items():
        ax[0].plot(hist['val_accuracy'], label=f'{name} (Val)')
        ax[1].plot(hist['val_loss'], label=f'{name} (Val)')

    ax[0].set_title('Validation Accuracy over Epochs')
    ax[0].set_xlabel('Epoch')
    ax[0].set_ylabel('Accuracy')
    ax[0].legend()

    ax[1].set_title('Validation Loss over Epochs')
    ax[1].set_xlabel('Epoch')
    ax[1].set_ylabel('Loss')
    ax[1].legend()

    st.pyplot(fig)

    # --- 3. Sample Predictions ---
    st.divider()
    st.header("🖼️ Sample Predictions (Adam Model)")
    
    random_indices = np.random.choice(len(x_test), 10, replace=False)
    sample_images = x_test[random_indices]
    sample_labels = y_test[random_indices]
    
    predictions = trained_models['Adam'].predict(sample_images)
    pred_labels = np.argmax(predictions, axis=1)

    fig2, axes = plt.subplots(2, 5, figsize=(12, 6))
    for i, ax_img in enumerate(axes.flat):
        ax_img.imshow(sample_images[i].squeeze(), cmap='gray')
        color = 'green' if pred_labels[i] == sample_labels[i] else 'red'
        ax_img.set_title(f"P: {pred_labels[i]} | A: {sample_labels[i]}", color=color, fontsize=10)
        ax_img.axis('off')
    
    st.pyplot(fig2)

else:
    st.info("Adjust settings in the sidebar and click the button to start.")