# Face Recognition Using Tensorflow
This is a re-implementation of the face recognizer in [FaceNet](https://github.com/davidsandberg/facenet).
I have resealed the code. If you are a Virgo, you will understand that why I spend time doing this work. <br>
## Face Detection
If you want to detect one image:
```bash
python main.py detect
```
or
```bash
python main.py detect --image your_image_path.jpg
```
If you want to detect video:
```bash
python main.py detecting
```
## Make Align Face
If you want to detect and align all the faces :
```bash
python main.py all --input_dir your_input_dir_path \
                   --output_dir your_output_dir_path
```
If you want to make single align face from your database:
```bash
python main.py single --input_dir your_input_dir_path \
                      --output_dir your_output_dir_path
```
Don't forget to revise the nrof_classes(in configs/config.json) to be your number of classes, otherwise "AssertionError: number of classes should be respect to fact" will occur
##Valid LFW
Before this, you need to download the pretrained model, such as the [model of CASIA-WebFace](https://drive.google.com/open?id=1R77HmFADxe87GmoLwzfgMu_HY0IhcyBz)
and revise the valid_model_path(in configs/lfw_config.json) to be your model path.
```bash
python main.py valid --input_dir your_input_path
```
## Compare face
Before this, you need to download the pretrained model, such as the [model of CASIA-WebFace](https://drive.google.com/open?id=1R77HmFADxe87GmoLwzfgMu_HY0IhcyBz) <br>
and revise the valid_model_path(in configs/lfw_config.json) to be your model path. <br>
If you want just to have a test, run
```bash
python main.py compare
```
If you want to compare your own images:
```bash
python main.py compare --compared your_image_path1.jpg your_image_path2.jpg
```
## Real Time Recognition
Before this, you need to download the pretrained model, such as the [model of CASIA-WebFace](https://drive.google.com/open?id=1R77HmFADxe87GmoLwzfgMu_HY0IhcyBz) <br>
and revise the valid_model_path(in configs/lfw_config.json) to be your model path. <br>
Then, you should make your classifier:
```bash
python main.py classifier  --input_dir your_input_path
```
Next, you can recognize the people by:
```bash
python main.py recognize --input_dir your_input_path
```
Note that the two "your_input_path" should be the same.
## Train your model
You should first make align your database
```bash
python main.py single --input_dir your_input_dir_path_for_training \
                      --output_dir your_output_dir_path_for_saving
```
Then you can train your model by
```bash
python main.py train  --input_dir your_input_dir_path
```
Note that your_input_dir_path should be equal to your_output_dir_path_for_saving





