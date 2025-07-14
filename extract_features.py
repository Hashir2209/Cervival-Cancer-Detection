import argparse
from sklearn.preprocessing import LabelEncoder
from keras.applications import ResNet50, VGG16
from keras.applications import imagenet_utils
from keras.preprocessing.image import img_to_array, load_img
from imutils import paths
import numpy as np
import pickle
import os

def extract_features(model_name, root_dir="data/"):
    # Model selection
    model_dict = {
        'resnet50': ResNet50(weights="imagenet", include_top=False),
        'vgg16': VGG16(weights="imagenet", include_top=False)
    }
    
    if model_name.lower() not in model_dict:
        raise ValueError(f"Unsupported model: {model_name}. Choose from {list(model_dict.keys())}")
    
    model = model_dict[model_name.lower()]
    le = None

    # Create output directory
    os.makedirs("ResNet50_output", exist_ok=True)

    # Process data splits
    for split in ("data", "dummy"):
        p = os.path.sep.join([root_dir, split])
        imagePaths = list(paths.list_images(p))

        # Extract class labels
        labels = [p.split(os.path.sep)[-2] for p in imagePaths]

        # Initialize label encoder
        if le is None:
            le = LabelEncoder()
            le.fit(labels)

        # Output CSV path
        csvPath = os.path.sep.join(["ResNet50_output", f"{model_name}_{split}.csv"])
        
        with open(csvPath, "w") as csv:
            # Process images in batches
            for (b, i) in enumerate(range(0, len(imagePaths), 32)):
                print(f"[INFO] Processing batch {b + 1}/{int(np.ceil(len(imagePaths) / 32))}")
                
                batchPaths = imagePaths[i:i + 32]
                batchLabels = le.transform(labels[i:i + 32])
                batchImages = []

                # Preprocess images
                for imagePath in batchPaths:
                    image = load_img(imagePath, target_size=(224, 224))
                    image = img_to_array(image)
                    image = np.expand_dims(image, axis=0)
                    image = imagenet_utils.preprocess_input(image)
                    batchImages.append(image)

                # Extract features
                batchImages = np.vstack(batchImages)
                features = model.predict(batchImages, batch_size=32)
                features = features.reshape((features.shape[0], -1))
        
                # Write features to CSV
                for (label, vec) in zip(batchLabels, features):
                    vec = ",".join([str(v) for v in vec])
                    csv.write(f"{label},{vec}\n")

    # Serialize label encoder
    with open(f"ResNet50_output/{model_name}_le.cpickle", "wb") as f:
        pickle.dump(le, f)

def main():
    parser = argparse.ArgumentParser(description='Extract features from images using pre-trained models')
    parser.add_argument('--models', nargs='+', default=['resnet50'], 
                        help='List of models to use for feature extraction')
    parser.add_argument('--root', type=str, default="data/", 
                        help='Root directory of data')
    
    args = parser.parse_args()
    
    for model in args.models:
        print(f"[INFO] Extracting features using {model}")
        extract_features(model, args.root)

if __name__ == "__main__":
    main()
