model_deploy.py \
	--mlir vit.mlir \
	--quantize F32 \
 	--chip bm1684x \
	--test_input vit_in_f32.npz \
	--test_reference vit_top_outputs.npz \
	--tolerance 0.85,0.85 \
 	--model vit_1684x_f32.bmodel

