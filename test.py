
# windows
set nnUNet_preprocessed=D:/project/nnUNet/nnUNetFrame/DATASET/nnUNet_preprocessed
set nnUNet_raw=D:/project/nnUNet/nnUNetFrame/DATASET/nnUNet_raw
set nnUNet_results=D:/project/nnUNet/nnUNetFrame/DATASET/nnUNet_results

# linux
export nnUNet_raw="/home/intern/nnUNet/nnUNetFrame/DATASET/nnUNet_raw"
export nnUNet_preprocessed="/home/intern/nnUNet/nnUNetFrame/DATASET/nnUNet_preprocessed"
export nnUNet_results="/home/intern/nnUNet/nnUNetFrame/DATASET/nnUNet_results"
export CUDA_VISIBLE_DEVICES=2

nnUNetv2_plan_and_preprocess -d 001 --verify_dataset_integrity

nnUNetv2_train 001 2d 0

export CUDA_VISIBLE_DEVICES=2
echo $CUDA_VISIBLE_DEVICES

nnUNetv2_train 001 2d 0
nnUNetv2_train 001 2d 1
nnUNetv2_train 001 2d 2
nnUNetv2_train 001 2d 3
nnUNetv2_train 001 2d 4

nnUNetv2_predict -i /home/intern/nnUNet/nnUNetFrame/DATASET/nnUNet_raw/Dataset001_Rectum/imagesTs -o /home/intern/nnUNet/nnUNetFrame/DATASET/nnUNet_results/Dataset001_Rectum/nnUNetTrainer__nnUNetPlans__2d/fold_3/testResult -d 001 -c 2d -f 3
nnUNetv2_predict -i INPUT_FOLDER -o OUTPUT_FOLDER -d DATASET_NAME_OR_ID -c CONFIGURATION
