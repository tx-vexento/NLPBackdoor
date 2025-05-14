from openai import OpenAI
import json
from json_repair import repair_json
import requests


class LLMAPI:
    def __init__(self):
        self.client = OpenAI(
            api_key="",
            base_url="https://api.siliconflow.cn/v1",
        )

        with open(
            "./model.json", "r"
        ) as f:
            self.model_map = json.loads(f.read())

        self.gpt_headers = {
            "Content-Type": "application/json",
            "api-key": "",
        }

    def call(
        self,
        messages=[{"role": "user", "content": "Hello!"}],
        llm_name="qwen2.5-7b",
    ):
        if "gpt" in llm_name:
            gpt_url = f"https://gpt-wang.openai.azure.com/openai/deployments/{llm_name}/chat/completions?api-version=2024-02-15-preview"
            payload = {
                "messages": messages,
                "temperature": 0.1,
                "top_p": 0.95,
                "max_tokens": 800,
            }
            response = requests.post(
                gpt_url, headers=self.gpt_headers, data=json.dumps(payload)
            )

            resp = response.json()["choices"][0]["message"]["content"]
        else:
            resp = self.client.chat.completions.create(
                model=self.model_map[llm_name],
                messages=messages,
            )
            resp = resp.choices[0].message.content
        return resp

    def json_call(
        self, messages=[{"role": "user", "content": "Hello!"}], llm_name="qwen2.5-7b"
    ):
        if "gpt" in llm_name:
            gpt_url = f"https://gpt-wang.openai.azure.com/openai/deployments/{llm_name}/chat/completions?api-version=2024-02-15-preview"
            payload = {
                "messages": messages,
                "temperature": 0.1,
                "top_p": 0.95,
                "max_tokens": 800,
            }
            response = requests.post(
                gpt_url, headers=self.gpt_headers, data=json.dumps(payload)
            )

            resp_str = response.json()["choices"][0]["message"]["content"]
            resp_json = json.loads(resp_str)
            try:
                resp_json = json.loads(resp_str)
            except:
                try:
                    resp_json = json.loads(repair_json(resp_str))
                except:
                    print(f"bad resp_str: {resp_str}")
                    print(f"repair resp_str: {repair_json(resp_str)}")
                    # exit(0)
        else:
            resp = self.client.chat.completions.create(
                model=self.model_map[llm_name],
                messages=messages,
            )
            resp_str = resp.choices[0].message.content
            try:
                resp_json = json.loads(resp_str)
            except:
                try:
                    resp_json = json.loads(repair_json(resp_str))
                except:
                    print(f"bad resp_str: {resp_str}")
                    print(f"repair resp_str: {repair_json(resp_str)}")
                    # exit(0)
        return resp_json


if __name__ == "__main__":
    llm_api = LLMAPI()
    print(llm_api.call(llm_name="gpt35-16k"))
