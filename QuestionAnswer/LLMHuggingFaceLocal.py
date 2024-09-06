# Copyright 2024-, RGBYang.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM, AutoTokenizer
from langchain.llms.base import LLM

# 使用本地模型初始化LLM。
class LLMHuggingFaceLocal(LLM):
    def _call(self, prompt: str, stop: list = None) -> str:
        model_name = "qwen/Qwen2-0.5B"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)
        model.eval()  # 确保模型处于评估模式
        inputs = tokenizer(prompt, return_tensors="pt")
        outputs = model.generate(**inputs, max_length=64)
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response
    
    @property
    def _llm_type(self) -> str:
        return "local_huggingface_causallm"