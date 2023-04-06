#/bin/bash

set -eo pipefail

if [[ -z "$TT_METAL_HOME" ]]; then
  echo "Must provide TT_METAL_HOME in environment" 1>&2
  exit 1
fi

if [ "$TT_METAL_ENV" != "dev" ]; then
  echo "Must set TT_METAL_ENV as dev" 1>&2
  exit 1
fi

cd $TT_METAL_HOME

make clean
make build
make tests

source build/python_env/bin/activate
export PYTHONPATH=$TT_METAL_HOME
python -m pip install -r tests/python_api_testing/requirements.txt

env python tests/scripts/run_build_kernels_for_riscv.py -j 16
env python tests/scripts/run_llrt.py --skip-driver-tests
env ./build/test/llrt/test_silicon_driver
env python tests/scripts/run_tt_metal.py

./tests/scripts/run_python_api_unit_tests.sh
env pytest tests/python_api_testing/models/bert/bert_encoder.py -k bert_encoder
env pytest tests/python_api_testing/models/bert -k bert_question_and_answering

env pytest tests/python_api_testing/models/t5 -k t5_dense_act_dense
env pytest tests/python_api_testing/models/t5 -k t5_layer_norm
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

env pytest tests/python_api_testing/models/stable_diffusion -k residual_block
env pytest tests/python_api_testing/models/stable_diffusion/CLIP -k CLIPMLP
env pytest tests/python_api_testing/models/stable_diffusion/fused_ops -k feedforward
env pytest tests/python_api_testing/models/stable_diffusion/fused_ops -k silu
env pytest tests/python_api_testing/models/stable_diffusion/fused_ops -k up_and_down_block
