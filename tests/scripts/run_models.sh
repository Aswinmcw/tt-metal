#/bin/bash

set -eo pipefail

if [[ -z "$TT_METAL_HOME" ]]; then
  echo "Must provide TT_METAL_HOME in environment" 1>&2
  exit 1
fi

cd $TT_METAL_HOME

export PYTHONPATH=$TT_METAL_HOME

# Turning on fast dispatch for the remaining models
if [[ ! -z "$FAST_DISPATCH" ]]; then
    echo "Running models in fast dispatch mode"
else
    export TT_METAL_SLOW_DISPATCH_MODE=1
fi

env pytest tests/python_api_testing/models/whisper -k whisper_attention
env pytest tests/python_api_testing/models/whisper -k WhipserEncoderLayer_inference
env pytest tests/python_api_testing/models/whisper -k WhipserEncoder_inference
env pytest tests/python_api_testing/models/whisper -k WhipserDecoderLayer_inference
env pytest tests/python_api_testing/models/whisper -k WhipserDecoder_inference
env pytest tests/python_api_testing/models/whisper -k whisper_model
env pytest tests/python_api_testing/models/whisper -k whisper_for_audio_classification
env pytest tests/python_api_testing/models/whisper -k whisper_for_conditional_generation

env pytest tests/python_api_testing/models/stable_diffusion/tests/test_embedding.py

# Bad tests, don't enable: Hanging post commit 8/24/23 debug war room session, see PR#2297, PR#2301
# env pytest tests/python_api_testing/models/stable_diffusion/tests/test_cross_attn_down_block.py -k test_run_cross_attn_down_block_real_input_inference
# env pytest tests/python_api_testing/models/stable_diffusion/tests/test_cross_attn_up_block.py -k test_run_cross_attn_up_block_real_input_inference
# env pytest tests/python_api_testing/models/stable_diffusion/tests/test_downblock_2d.py -k test_run_downblock_real_input_inference
# env pytest tests/python_api_testing/models/stable_diffusion/tests/test_unet_mid_block.py -k test_run_unet_mid_block_real_input_inference
# env pytest tests/python_api_testing/models/stable_diffusion/tests/test_upblock_2d.py -k test_run_upblock_real_input_inference
# env pytest tests/python_api_testing/models/stable_diffusion/tests -k test_unbatched_stable_diffusion

env pytest tests/python_api_testing/models/deit/tests/test_deit_for_image_classification_with_teacher.py -k test_deit_for_image_classification_with_teacher_inference

env pytest tests/python_api_testing/models/vit/tests/test_vit_image_classification.py -k test_vit_image_classification

env pytest tests/python_api_testing/models/metal_BERT_large_15/test_bert_batch_dram.py::test_bert_batch_dram
env pytest tests/python_api_testing/models/metal_BERT_large_15/test_bert_batch_dram.py::test_bert_batch_dram_with_program_cache

env pytest tests/python_api_testing/models/t5 -k t5_dense_act_dense
env pytest tests/python_api_testing/models/t5 -k t5_layer_norm
env pytest tests/python_api_testing/models/t5 -k t5_attention
env pytest tests/python_api_testing/models/t5 -k t5_layer_ff
env pytest tests/python_api_testing/models/t5 -k t5_layer_self_attention
env pytest tests/python_api_testing/models/t5 -k t5_layer_cross_attention
env pytest tests/python_api_testing/models/t5 -k t5_block
env pytest tests/python_api_testing/models/t5 -k t5_stack
env pytest tests/python_api_testing/models/t5 -k t5_model

env pytest tests/python_api_testing/models/synthetic_gradients -k batchnorm1d_test
env pytest tests/python_api_testing/models/synthetic_gradients -k linear_test
env pytest tests/python_api_testing/models/synthetic_gradients -k block_test
env pytest tests/python_api_testing/models/synthetic_gradients -k full_inference

env pytest tests/python_api_testing/models/llama -k llama_layer_norm
env pytest tests/python_api_testing/models/llama -k llama_mlp
env pytest tests/python_api_testing/models/llama -k llama_attention
env pytest tests/python_api_testing/models/llama -k llama_decoder

env pytest tests/python_api_testing/models/falcon/tests/test_falcon_causallm.py::test_FalconCausalLM_inference[BFLOAT16-DRAM-falcon_7b-layers_32-batch1_seqlen128]

env pytest tests/python_api_testing/models/lenet -k test_lenet_inference
env pytest tests/python_api_testing/models/ConvNet_MNIST/tests -k mnist_inference
env pytest tests/python_api_testing/models/mnist/tests -k mnist_inference

env pytest tests/python_api_testing/models/roberta -k roberta_self_attention
env pytest tests/python_api_testing/models/roberta -k roberta_self_output
env pytest tests/python_api_testing/models/roberta -k roberta_attention
env pytest tests/python_api_testing/models/roberta -k roberta_intermediate
env pytest tests/python_api_testing/models/roberta -k roberta_output
env pytest tests/python_api_testing/models/roberta -k roberta_layer
env pytest tests/python_api_testing/models/roberta -k roberta_encoder
env pytest tests/python_api_testing/models/roberta -k roberta_pooler
env pytest tests/python_api_testing/models/roberta -k roberta_model
env pytest tests/python_api_testing/models/roberta -k roberta_lm_head
env pytest tests/python_api_testing/models/roberta -k roberta_for_masked_lm
env pytest tests/python_api_testing/models/roberta -k roberta_for_qa
env pytest tests/python_api_testing/models/roberta -k roberta_for_sequence_classification
env pytest tests/python_api_testing/models/roberta -k roberta_classification_head
env pytest tests/python_api_testing/models/roberta -k roberta_for_token_classification
env pytest tests/python_api_testing/models/roberta -k roberta_for_multiple_choice

env pytest tests/python_api_testing/models/yolov5/tests -k Yolov5_detection_model

env pytest tests/python_api_testing/models/bloom -k baddbmm
env pytest tests/python_api_testing/models/bloom -k bloom_attention
env pytest tests/python_api_testing/models/bloom -k bloom_block
env pytest tests/python_api_testing/models/bloom -k bloom_gelu_forward
env pytest tests/python_api_testing/models/bloom -k bloom_merge_heads
env pytest tests/python_api_testing/models/bloom -k bloom_mlp
env pytest tests/python_api_testing/models/bloom -k bloom_model
env pytest tests/python_api_testing/models/bloom -k bloom_causal_lm

env pytest tests/python_api_testing/models/yolov3 -k conv2d_module
env pytest tests/python_api_testing/models/yolov3 -k conv_module
env pytest tests/python_api_testing/models/yolov3 -k concat_module
env pytest tests/python_api_testing/models/yolov3 -k bottleneck_module
env pytest tests/python_api_testing/models/yolov3 -k detect_module
env pytest tests/python_api_testing/models/yolov3 -k detection_model
env pytest tests/python_api_testing/models/yolov3 -k upsample_module

env pytest tests/python_api_testing/models/EfficientNet/tests -k efficientnet_b0_model_real
env pytest tests/python_api_testing/models/EfficientNet/tests -k efficientnet_v2_s_model_real
env pytest tests/python_api_testing/models/EfficientNet/tests -k efficientnet_lite0_model_real

env pytest tests/python_api_testing/models/nanogpt -k nanogpt_model_real
