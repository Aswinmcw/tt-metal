import sys
import torch
from pathlib import Path
from loguru import logger

f = f"{Path(__file__).parent}"
sys.path.append(f"{f}/..")
sys.path.append(f"{f}/../torch")
sys.path.append(f"{f}/../..")
sys.path.append(f"{f}/../../..")
sys.path.append(f"{f}/../../../..")

from modeling_squeezebert import PytorchSqueezeBertForQuestionAnswering
from utils import comp_outputs, get_answer

from transformers import (
    SqueezeBertForQuestionAnswering as HF_SqueezeBertForQuestionAnswering,
)
from transformers import AutoTokenizer

from utility_functions_new import (
    profiler,
    enable_compile_cache,
    disable_compile_cache,
    comp_pcc,
)


def test_squeezebert_qa_inference():
    tokenizer = AutoTokenizer.from_pretrained("squeezebert/squeezebert-uncased")
    HF_model = HF_SqueezeBertForQuestionAnswering.from_pretrained(
        "squeezebert/squeezebert-uncased"
    )

    state_dict = HF_model.state_dict()
    config = HF_model.config
    get_head_mask = HF_model.transformer.get_head_mask
    get_extended_attention_mask = HF_model.transformer.get_extended_attention_mask

    PT_model = PytorchSqueezeBertForQuestionAnswering(config)
    res = PT_model.load_state_dict(state_dict)
    PT_model.transformer.get_extended_attention_mask = get_extended_attention_mask
    PT_model.transformer.get_head_mask = get_head_mask

    question, context = (
        "Where do I live?",
        "My name is Merve and I live in İstanbul.",
    )

    inputs = tokenizer(question, context, return_tensors="pt")

    disable_compile_cache()
     
    with torch.no_grad():
        profiler.enable()
        profiler.start("\nExec time of reference model")
        HF_output = HF_model(**inputs)
        profiler.end("\nExec time of reference model")
        
        profiler.start("\nExecution time of tt_SqueezeBert first run")
        torch_output = PT_model(**inputs)
        profiler.end("\nExecution time of tt_SqueezeBert first run") 
        enable_compile_cache()
        
        PERF_CNT = 5 
        for i in range(PERF_CNT):
            profiler.start("\nAverage execution time of tt_SqueezeBert model")
            torch_output = PT_model(**inputs) 
            profiler.end("\nAverage execution time of tt_SqueezeBert model")

    torch_answer = get_answer(inputs, HF_output, tokenizer)
    pt_answer = get_answer(inputs, torch_output, tokenizer)

    logger.info("HF Model answered")
    logger.info(torch_answer)

    logger.info("PT Model answered")
    logger.info(pt_answer)

    does_pass_1, does_pass_2 = comp_outputs(HF_output, torch_output)

    if does_pass_1 and does_pass_2:
        logger.info("squeezebertForQA Passed!")
    else:
        logger.warning("squeezebertForQA Failed!")

    assert does_pass_1 and does_pass_2
    
    profiler.print()

if __name__ == "__main__":
    test_squeezebert_qa_inference()
