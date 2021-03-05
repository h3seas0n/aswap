# ASWAP: A Fully Autonomous Subaquatic Waste Acquiring and Preprocessing System

Smart solution to collect waster underwater and sort it automatically.

## Introduction

Using a weakly supervised object detection model aboard a versatile drone, we can locate and collect waste. We use advanced classification models to sort garbage in our docking stations so that it can be recycled and resold.

## Requirements

* Python Version > 3.5
* Requirements.txt

## Datasets

In this project we used 4 datasets: The [trashnet](https://github.com/garythung/trashnet) dataset, arkadiyhack's [drinking](https://www.kaggle.com/arkadiyhacks/drinking-waste-classification) waste dataset, another [waste](https://www.kaggle.com/szdxfkmgnb/waste-classification) dataset, and the [JAMSTEC](http://www.godac.jamstec.go.jp/catalog/dsdebris/e/index.html) underwater debris dataset. 

## Installation

You can use the packagemanager [pip](https://pip.pypa.io/en/stable/) to install the required dependencies.

```bash
pip install -r requirements.txt
```

or 

```bash
pip3 install -r requirements.txt
```

## Usage

# Step 1: Fetch data:

* Detection model: run detch_dataset.py
* Classification model: included in GarbageClassification.ipynb:

    ```bash
    #Download and unzip datasets:
    !kaggle datasets download -d szdxfkmgnb/waste-classification
    !kaggle datasets download -d arkadiyhacks/drinking-waste-classification
    # Upload trashnet-dataset-resized from local: https://drive.google.com/drive/folders/0B3P9oO5A3RvSUW9qTG11Ul83TEE 

    #unzip and delete .zip files
    !unzip \*.zip  && rm *.zip
    ```

# Step 2: Prepare the fetched data:
* Write data to csv and split datasets:

    ```python
    import os
    import csv
    from sklearn.model_selection import train_test_split

    def append_list_as_row(filename, list_of_elem):
        with open(filename, 'a+', newline='') as write_obj:
            csv_writer = csv.writer(write_obj)
            csv_writer.writerow(list_of_elem)

    def create_csv(directory, csv, labels):
        for list in os.listdir(directory):
            for img in os.listdir(os.path.join(directory, list)):
                img_name = os.path.basename(img)
                append_list_as_row(csv, [img_name, labels[list]])

    #Create and add header to csv files:
    append_list_as_row('labels1.csv', ['id', 'class'])
    #...

    #Define paths:
    dataset1 = os.path.join(current_dir, 'dataset')
    #...

    #Create csv files for the three datasets:
    create_csv(dataset1, 'labels1.csv', labels)
    #...

    train_filenames1, test_filenames1, train_labels1, test_labels1 = train_test_split(img_names1, img_labels1, train_size=0.8,random_state=42)

    ```
    ...

* Create subdirectories and move data:

    ```python
        import shutil

        def splitData(filenames, labels, dest_dir, src_dir, label_dict, names_dict=0):
            for item_name, label in zip(filenames, labels):
                item_name = os.path.basename(item_name)
                if names_dict == 0:
                img_type = label_dict.get(str(label))
                img_name = img_type
                else:
                img_type = ''.join([i for i in item_name if not i.isdigit()]) 
                img_type =  os.path.splitext(img_type)[0].replace(',', '')
                img_name = label_dict.get(str(label))
            
                train_path = src_dir + '/' + img_type + '/' + item_name
                shutil.copy(train_path, dest_dir + '/' + img_name)

        splitData(train_filenames1, train_labels1, train1_dest_dir, dataset1, labels_1)
    ```
    ...

* Apply data augmentation:

    ```python
        from tensorflow.keras.preprocessing.image import ImageDataGenerator
        
        train_datagen = ImageDataGenerator(rescale = 1.0/255.0, rotation_range=40, width_shift_range=0.2, height_shift_range=0.2, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
        test_datagen = ImageDataGenerator(rescale = 1.0/255.0)
    ```

# Step 3: Feed and train the models:

* Create data-batches:

    ```python
        batch_size = 20

        train_gen = train_datagen.flow_from_directory(train1_dest_dir, target_size=(255, 255), batch_size=batch_size, class_mode='binary')
        #...
    ```

* Transfer train both models:

    ```python

        from tensorflow.keras.applications import ResNet50V2

        img_size = 255

        model = ResNet50V2(input_shape = (img_size, img_size,3), include_top=False, weights='imagenet')
    ```
    Same procedure for detection model: Xception

    * Make layers untrainable and add last layers - Output, Flatten, Relu- and softmax-Dense, Dropout layers:

        ```python
            for layer in model.layers: 
                layer.trainable = False

            x = model.output
            x = Flatten()(x)
            x = Dense(1024, activation='relu')(x)
            x = Dropout(0.2)(x)
            pred = Dense(4, activation='softmax')(x)
        ```

    * Combine, compile and train transfer-trained ResNet/Xception model:

        ```python
            from tensorflow.keras.models import Model
            from tensorflow.keras.models import load_model
            from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, Flatte

            model = Model(inputs=model.input, outputs=pred)

            model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy']) 
            model.fit(train_gen, validation_data=test_gen, validation_steps = 25, epochs=50, steps_per_epoch=50, callbacks=[tensorboard_callback])
            model.save('model_classification.h5')
        ```

# Step 4: View the results and retrain - Classification model:

* Classification - Retrain on third dataset to improve False negatives/positives:

    ```python
        train5_gen = train_datagen.flow_from_directory(dataset1, target_size=(255, 255), batch_size=batch_size, class_mode='binary')
        model.fit(train5_gen, epochs=50, steps_per_epoch=50)
        model.save('model_classification_retrained.h5')
    ```

# Step 5: View the final results and benchmark:

* Test on single image:
    
    * Classification model:

        ```python
            import matplotlib.pyplot as plt
            
            model = load_model('model_classification_retrained.h5')
            def visualize(img_path) 
                img = image.load_img(img_path, target_size=(255, 255))
                img = image.img_to_array(img)
                img = np.expand_dims(img, axis=0)
                img = img/255
                plt.imshow(img[0])
                plt.axis('on')
                plt.show()

                pred = np.argmax(model.predict(img))
                print(labels_1.get(str(pred)))
        ```

    * Detection model: 
    utilsDetection\detection.py

* Test model on dataset - live accuracy - only for classification:

    ```python
        def test_live_accuracy(dataset):
            acc = 0
            i = 0
            incorrect = []
            for waste_type in os.listdir(dataset)[1:]:
                print('///////////////////////')
                print('///////////////////////')
                print('Now testing: ' + waste_type)
                print('///////////////////////')
                print('///////////////////////')
                for img in os.listdir('dataset/' + waste_type):
                    i += 1
                    img = image.load_img('dataset/' + waste_type + '/' + img, target_size=(255, 255))
                    img = image.img_to_array(img)
                    img = np.expand_dims(img, axis=0)
                    img = img/255
                    pred = np.argmax(model.predict(img))
                    pred = labels_1.get(str(pred))
                    if pred == waste_type:
                    acc += 1
                    else:
                    x = waste_type + ' - ' + pred
                    incorrect.append(x)
                    print('Current Accuracy: ' + str(acc/i))
    ```
    
    
# Visit our website!: https://aquation.ml
