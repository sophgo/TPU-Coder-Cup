model_transform.py \
	--model_name vit \
	--model_def vit-base-patch16-224.onnx \
	--input_shapes [[1,3,224,224]] \
	--mean 127.5,127.5,127.5 \
	--scale 0.00784313725490196,0.00784313725490196,0.00784313725490196 \
	--keep_aspect_ratio \
	--pixel_format rgb \
	--test_input ../model_resnet18/images/dog.jpg \
	--test_result vit_top_outputs.npz \
	--mlir vit.mlir \

