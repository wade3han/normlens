import logging
import os
import time
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List

import openai


class ChatLanguageModel(ABC):
    def __init__(self,
                 engine: str,
                 device: str = "",
                 temperatue=0.1,
                 topp=0.95,
                 frequency_penalty=0.0,
                 presence_penalty=0.0):
        self.device = device
        self.engine = engine
        self.temperature = temperatue
        self.topp = topp
        self.frequency_penalty = frequency_penalty
        self.presence_penalty = presence_penalty
        self.chat_memory = []
        self.system_prompt = None

    def set_system_prompt(self, prompt: str) -> None:
        self.system_prompt = prompt

    @abstractmethod
    def get_template_based_responses(self,
                                     conversation_template: List[str],
                                     input_informations: List[Dict[str, Any]]) -> List[List[Optional[str]]]:
        raise NotImplementedError


class OpenaiChatGpt(ChatLanguageModel):
    def __init__(self, engine: str, device: str = "", temperatue=0.1, topp=0.95, frequency_penalty=0.0,
                 presence_penalty=0.0):
        openai.api_key = os.environ["OPENAI_API_KEY"]
        super().__init__(engine, device, temperatue, topp, frequency_penalty, presence_penalty)

    def _get_chat_messages(self, context: str) -> List[Dict[str, Any]]:
        if self.system_prompt is None:
            return self.chat_memory + [{'role': 'user', 'content': context}]
        else:
            return [{'role': 'system', 'content': self.system_prompt}] \
                + self.chat_memory + [{'role': 'user', 'content': context}]

    def clear_chat_memory(self) -> None:
        self.chat_memory = []

    def create_response(self, content: str) -> Optional[str]:
        # Retry logic --- 10 times
        for _ in range(10):
            try:
                messages = self._get_chat_messages(content)
                response = self._create_response_chat(messages)
                self.chat_memory.append({'role': 'user', 'content': content})
                self.chat_memory.append({'role': 'assistant', 'content': response})

            except openai.error.RateLimitError as e:
                print(f"Reach rate limit: {e}")
                time.sleep(30)
                continue
            except Exception as e:
                print(f"Exception: {e}")
                time.sleep(30)
                continue
            return response

        return None

    def _create_response_chat(self, messages: List[Dict[str, Any]]) -> Optional[str]:
        response = openai.ChatCompletion.create(
            model=self.engine,
            messages=messages,
            temperature=self.temperature,
            max_tokens=256,
            top_p=self.topp,
            frequency_penalty=self.frequency_penalty,
            presence_penalty=self.presence_penalty,
        )
        if response is None:
            return None

        return response['choices'][0]['message']['content']

    def get_template_based_responses(self,
                                     conversation_template: List[str],
                                     input_informations: List[Dict[str, Any]]) -> List[List[Optional[str]]]:
        if len(self.chat_memory) > 0:
            logging.warning('Chat memory is not empty, we clear it before starting a new conversation.')
            self.clear_chat_memory()

        all_responses = []
        for input_information in input_informations:
            responses = []
            for num_turn in range(len(conversation_template)):
                content = conversation_template[num_turn].format(**input_information)
                response = self.create_response(content)
                responses.append(response)
                time.sleep(1)
            self.clear_chat_memory()
            all_responses.append(responses)
        return all_responses


class OpenaiGeneralGpt:
    def __init__(self, engine: str, temperatue=0.1, topp=0.95, frequency_penalty=0.0,
                 presence_penalty=0.0):
        self.engine = engine
        self.temperature = temperatue
        self.topp = topp
        self.frequency_penalty = frequency_penalty
        self.presence_penalty = presence_penalty
        self.chat_memory = []
        self.system_prompt = None

    def _create_response_completion(self, content: str) -> Optional[str]:
        response = openai.Completion.create(
            engine=self.engine,
            prompt=content,
            temperature=self.temperature,
            max_tokens=256,
            top_p=self.topp,
            frequency_penalty=self.frequency_penalty,
            presence_penalty=self.presence_penalty,
            logprobs=1,
            stop=["\n\n"]
        )
        if response is None:
            return None

        return response['choices'][0]['text']

    def create_response(self, content: str) -> Optional[str]:
        # Retry logic --- 10 times
        for _ in range(10):
            try:
                response = self._create_response_completion(content)

            except openai.error.RateLimitError as e:
                print(f"Reach rate limit: {e}")
                time.sleep(30)
                continue
            except Exception as e:
                print(f"Exception: {e}")
                time.sleep(30)
                continue
            return response

        return None
