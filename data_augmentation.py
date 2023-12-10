# standard libraries
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array, array_to_img
from shutil import copyfile

try:
    # Load CSV file
    csv_file = r'train.csv'
    df = pd.read_csv(csv_file)
    print(df.head())
    print('\n')
    print("Shape of data:", df.shape)
    print("*******************************************")
    print(df['label'].value_counts())

    # Define a mapping dictionary for damage types
    damage_type_mapping = {
        1: 'crack',
        2: 'scratch',
        3: 'tire flat',
        4: 'dent',
        5: 'glass shatter',
        6: 'lamp broken'
    }

    # Add a new column "damage_type" based on the mapping
    df['damage_type'] = df['label'].map(damage_type_mapping)

    # Split the dataset into training and validation sets
    train_df, val_df = train_test_split(df, test_size=0.2, stratify=df['label'], random_state=42)

    # Define ImageDataGenerator for augmentation
    datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    # Output directory for augmented images
    output_directory = 'augmented_images'
    os.makedirs(output_directory, exist_ok=True)

    # Copy original images to augmentation folder and add "image_path" column
    train_df['image_path'] = ""
    for index, row in train_df.iterrows():
        try:
            original_img_path = os.path.join('D:/End To End ML/hackhathon/train/', row['filename'])
            new_img_path = os.path.join(output_directory, row['filename'])
            copyfile(original_img_path, new_img_path)
            train_df.at[index, 'image_path'] = new_img_path
        except Exception as e:
            print(f"Error copying file {row['filename']}: {e}")

    # Calculate the maximum total count for each class
    max_total_count = 10000

    # Function to generate augmented data for a given class
    def generate_augmented_data(df, datagen, class_label, max_total_count):
        augmented_data = []
        class_data = df[df['label'] == class_label]

        # Calculate the existing count for the class
        existing_count = len(class_data)

        # Calculate the required augmentation factor
        augmentation_factor = max(1, (max_total_count - existing_count) // existing_count)

        for index, row in class_data.iterrows():
            try:
                img = load_img(row['image_path'], target_size=(224, 224))
                img_array = img_to_array(img)
                img_array = img_array / 255.0

                for i in range(augmentation_factor):
                    augmented_img = datagen.random_transform(img_array)
                    augmented_data.append({
                        'image_id': f"{row['filename']}",
                        'filename': f"{row['filename']}_aug_{i}.jpg",
                        'label': class_label,
                        'damage_type': damage_type_mapping[class_label],
                        'image_path': f"{output_directory}/{row['filename']}_aug_{i}.jpg",    
                    })
                    augmented_img_path = os.path.join(output_directory, f"{row['filename']}_aug_{i}.jpg")
                    array_to_img(augmented_img).save(augmented_img_path)
            except Exception as e:
                print(f"Error augmenting data for {row['filename']}: {e}")

        return augmented_data

    # Generate augmented data for each class and combine with the original training data
    augmented_data = []

    for class_label in train_df['label'].unique():
        try:
            augmented_data.extend(generate_augmented_data(train_df, datagen, class_label, max_total_count))
        except Exception as e:
            print(f"Error generating augmented data for class {class_label}: {e}")

    # Combine augmented data with the original training data
    augmented_df = pd.DataFrame(augmented_data)
    augmented_train_df = pd.concat([train_df, augmented_df], ignore_index=True)
    # Remove the 'image_path' column in-place
    augmented_train_df.drop(columns=['image_path'], inplace=True)

    print(augmented_train_df.head())
    print(augmented_train_df.shape) 
    print('********************************')
    print("Class distribution after augmentation:")
    print(augmented_train_df['label'].value_counts())

    # Save the augmented data to a new CSV file
    augmented_train_df.to_csv('augmented_train_data.csv', index=False)

except Exception as e:
    print(f"An error occurred: {e}")
