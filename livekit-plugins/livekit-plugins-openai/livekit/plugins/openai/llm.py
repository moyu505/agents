# Copyright 2023 LiveKit, Inc.
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

from __future__ import annotations

import asyncio
import base64
from dataclasses import dataclass
from typing import Any, Awaitable, MutableSet

import httpx
from livekit import rtc
from livekit.agents import llm, utils

import openai
from openai.types.chat import ChatCompletionChunk, ChatCompletionMessageParam
from openai.types.chat.chat_completion_chunk import Choice

from my_plugins.turing_logs.sls_aliyun import put_logs

from .log import logger
from .models import (
    ChatModels,
    GroqChatModels,
    OctoChatModels,
    PerplexityChatModels,
    TogetherChatModels,
)
from .utils import AsyncAzureADTokenProvider


@dataclass
class LLMOptions:
    model: str | ChatModels


class LLM(llm.LLM):
    def __init__(
        self,
        *,
        model: str | ChatModels = "gpt-4o",
        api_key: str | None = None,
        base_url: str | None = None,
        client: openai.AsyncClient | None = None,
    ) -> None:
        self._opts = LLMOptions(model=model)
        self._client = client or openai.AsyncClient(
            api_key=api_key,
            base_url=base_url,
            http_client=httpx.AsyncClient(
                timeout=5.0,
                follow_redirects=True,
                limits=httpx.Limits(
                    max_connections=1000,
                    max_keepalive_connections=100,
                    keepalive_expiry=120,
                ),
            ),
        )
        self._running_fncs: MutableSet[asyncio.Task[Any]] = set()
        self._img_list = []
        self.MAXIMAGE = 1

    @staticmethod
    def with_azure(
        *,
        model: str | ChatModels = "gpt-4o",
        azure_endpoint: str | None = None,
        azure_deployment: str | None = None,
        api_version: str | None = None,
        api_key: str | None = None,
        azure_ad_token: str | None = None,
        azure_ad_token_provider: AsyncAzureADTokenProvider | None = None,
        organization: str | None = None,
        project: str | None = None,
        base_url: str | None = None,
    ) -> LLM:
        """
        This automatically infers the following arguments from their corresponding environment variables if they are not provided:
        - `api_key` from `AZURE_OPENAI_API_KEY`
        - `organization` from `OPENAI_ORG_ID`
        - `project` from `OPENAI_PROJECT_ID`
        - `azure_ad_token` from `AZURE_OPENAI_AD_TOKEN`
        - `api_version` from `OPENAI_API_VERSION`
        - `azure_endpoint` from `AZURE_OPENAI_ENDPOINT`
        """

        azure_client = openai.AsyncAzureOpenAI(
            azure_endpoint=azure_endpoint,
            azure_deployment=azure_deployment,
            api_version=api_version,
            api_key=api_key,
            azure_ad_token=azure_ad_token,
            azure_ad_token_provider=azure_ad_token_provider,
            organization=organization,
            project=project,
            base_url=base_url,
        )  # type: ignore

        return LLM(model=model, client=azure_client)

    @staticmethod
    def with_fireworks(
        *,
        model: str = "accounts/fireworks/models/llama-v3p1-70b-instruct",
        api_key: str | None = None,
        base_url: str | None = "https://api.fireworks.ai/inference/v1",
        client: openai.AsyncClient | None = None,
    ) -> LLM:
        return LLM(model=model, api_key=api_key, base_url=base_url, client=client)

    @staticmethod
    def with_groq(
        *,
        model: str | GroqChatModels = "llama3-8b-8192",
        api_key: str | None = None,
        base_url: str | None = "https://api.groq.com/openai/v1",
        client: openai.AsyncClient | None = None,
    ) -> LLM:
        return LLM(model=model, api_key=api_key, base_url=base_url, client=client)

    @staticmethod
    def with_octo(
        *,
        model: str | OctoChatModels = "llama-2-13b-chat",
        api_key: str | None = None,
        base_url: str | None = "https://text.octoai.run/v1",
        client: openai.AsyncClient | None = None,
    ) -> LLM:
        return LLM(model=model, api_key=api_key, base_url=base_url, client=client)

    @staticmethod
    def with_ollama(
        *,
        model: str = "llama3.1",
        base_url: str | None = "http://localhost:11434/v1",
        client: openai.AsyncClient | None = None,
    ) -> LLM:
        return LLM(model=model, api_key="ollama", base_url=base_url, client=client)

    @staticmethod
    def with_perplexity(
        *,
        model: str | PerplexityChatModels = "llama-3.1-sonar-small-128k-chat",
        api_key: str | None = None,
        base_url: str | None = "https://api.perplexity.ai",
        client: openai.AsyncClient | None = None,
    ) -> LLM:
        return LLM(model=model, api_key=api_key, base_url=base_url, client=client)

    @staticmethod
    def with_together(
        *,
        model: str | TogetherChatModels = "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
        api_key: str | None = None,
        base_url: str | None = "https://api.together.xyz/v1",
        client: openai.AsyncClient | None = None,
    ) -> LLM:
        return LLM(model=model, api_key=api_key, base_url=base_url, client=client)

    @staticmethod
    def create_azure_client(
        *,
        model: str | ChatModels = "gpt-4o",
        azure_endpoint: str | None = None,
        azure_deployment: str | None = None,
        api_version: str | None = None,
        api_key: str | None = None,
        azure_ad_token: str | None = None,
        azure_ad_token_provider: AsyncAzureADTokenProvider | None = None,
        organization: str | None = None,
        project: str | None = None,
        base_url: str | None = None,
    ) -> LLM:
        logger.warning("This alias is deprecated. Use LLM.with_azure() instead")
        return LLM.with_azure(
            model=model,
            azure_endpoint=azure_endpoint,
            api_version=api_version,
            api_key=api_key,
            azure_ad_token=azure_ad_token,
            azure_ad_token_provider=azure_ad_token_provider,
            organization=organization,
            project=project,
            base_url=base_url,
        )

    def push_video(self,frame: rtc.VideoFrame):
        # print('====push_video====')
        self._img_list.append(frame)
        if len(self._img_list) > self.MAXIMAGE:
            self._img_list.pop(0)
        pass
    def chat(
        self,
        *,
        chat_ctx: llm.ChatContext,
        fnc_ctx: llm.FunctionContext | None = None,
        temperature: float | None = 0.9,
        n: int | None = 1,
        parallel_tool_calls: bool | None = None,
    ) -> "LLMStream":
        opts: dict[str, Any] = dict()
        if fnc_ctx and len(fnc_ctx.ai_functions) > 0:
            fncs_desc = []
            for fnc in fnc_ctx.ai_functions.values():
                fncs_desc.append(llm._oai_api.build_oai_function_description(fnc))

            opts["tools"] = fncs_desc

            if fnc_ctx and parallel_tool_calls is not None:
                opts["parallel_tool_calls"] = parallel_tool_calls
        if len(self._img_list) >= 1:
            # 将最后一个messages加入图片
            print(f'存在图像=={len(self._img_list)}')
            
            messagesList = chat_ctx.messages
            len1 = len(messagesList)
            for i in range(len1):
                content = messagesList[i].content
                if isinstance(content, list):
                    # 删除图片
                    messagesList[i].content = content[0]
                    pass
                pass
                
            # if len(chat_ctx.messages) >= 4:
                
                # print(f'==inputStr2========{inputStr2}====================={type(inputStr2)}=====')
                # if isinstance(chat_ctx.messages[-3].content, str):
                #     print(f'==========xyxyxyxyxy=======================')
                # else:
                #     # 循环的删除 messages的图片
                #     len1 = len(inputStr2)
                #     for ii, inputStr in enumerate(inputStr2) :

                #         pass
                #     if isinstance(chat_ctx.messages[-3].content, list):
                #         for ii, inputStr in enumerate(inputStr2) :
                #             if isinstance(inputStr, str):
                #                 print(f'==========删除最后一个messages的图片==============')
                #                 chat_ctx.messages[-3].content = inputStr
                #                 break
                #     # chat_ctx.messages[-3].content = inputStr2
               
            # 加入图片
            img = llm.ChatImage(image=self._img_list[-1])
            inputStr = chat_ctx.messages[-1].content
            print(f'====inputStr======{inputStr}====================={type(inputStr)}=====')
            if isinstance(chat_ctx.messages[-1].content, str):
                
                chat_ctx.messages[-1].content = [inputStr, img]
            else:
                print(f'存在多个图像=={len(chat_ctx.messages[-1].content)}')
                chat_ctx.messages[-1].content.append(img)
            
            pass
        else:
            print(f'没有图像=={len(self._img_list)}')
        print('========llm1=================')
        chat_ctx.print()
        print('========llm2=================')
        # 控制上下文轮数
        if len(chat_ctx.messages) > 11:
            # chat_ctx.messages = chat_ctx.messages[:3] + chat_ctx.messages[5:]
            chat_ctx.messages = chat_ctx.messages[:0] + chat_ctx.messages[3:]
            pass
        print(f'===============上下文轮数=={len(chat_ctx.messages)}')
        messages = _build_oai_context(chat_ctx, id(self))
        
        cmp = self._client.chat.completions.create(
            messages=messages,
            model=self._opts.model,
            n=n,
            temperature=temperature,
            stream=True,
            **opts,
        )
        if len(self._img_list) >= self.MAXIMAGE:
            self._img_list.pop(0)

        return LLMStream(oai_stream=cmp, chat_ctx=chat_ctx, fnc_ctx=fnc_ctx,inputMessage=messages)


class LLMStream(llm.LLMStream):
    def __init__(
        self,
        *,
        oai_stream: Awaitable[openai.AsyncStream[ChatCompletionChunk]],
        chat_ctx: llm.ChatContext,
        fnc_ctx: llm.FunctionContext | None,
        inputMessage: None,
    ) -> None:
        super().__init__(chat_ctx=chat_ctx, fnc_ctx=fnc_ctx)
        self._awaitable_oai_stream = oai_stream
        self._oai_stream: openai.AsyncStream[ChatCompletionChunk] | None = None

        # current function call that we're waiting for full completion (args are streamed)
        self._tool_call_id: str | None = None
        self._fnc_name: str | None = None
        self._fnc_raw_arguments: str | None = None
        self.inputMessage = inputMessage
        self._final_res = ""

    async def aclose(self) -> None:
        if self._oai_stream:
            await self._oai_stream.close()
        print('self._final_res:',self._final_res )
        return await super().aclose()

    async def __anext__(self):
        if not self._oai_stream:
            self._oai_stream = await self._awaitable_oai_stream

        async for chunk in self._oai_stream:
            if chunk.choices[0].finish_reason == 'stop':
                # 结束上传日志
                logList = [{'input':self.inputMessage}, {'output':self._final_res}, {"roomID": self._chat_ctx.roomId}]
                put_logs(logList)

            for choice in chunk.choices:
                chat_chunk = self._parse_choice(choice)
                if chat_chunk is not None:
                    return chat_chunk
                pass
            

        raise StopAsyncIteration

    def _parse_choice(self, choice: Choice) -> llm.ChatChunk | None:
        delta = choice.delta

        if delta.tool_calls:
            # check if we have functions to calls
            for tool in delta.tool_calls:
                if not tool.function:
                    continue  # oai may add other tools in the future

                call_chunk = None
                if self._tool_call_id and tool.id and tool.id != self._tool_call_id:
                    call_chunk = self._try_run_function(choice)

                if tool.function.name:
                    self._tool_call_id = tool.id
                    self._fnc_name = tool.function.name
                    self._fnc_raw_arguments = tool.function.arguments or ""
                elif tool.function.arguments:
                    self._fnc_raw_arguments += tool.function.arguments  # type: ignore

                if call_chunk is not None:
                    return call_chunk

        if choice.finish_reason == "tool_calls":
            # we're done with the tool calls, run the last one
            return self._try_run_function(choice)
        
        if delta.content is not None:
            self._final_res += delta.content
            # print('===>', self._final_res)
        return llm.ChatChunk(
            choices=[
                llm.Choice(
                    delta=llm.ChoiceDelta(content=delta.content, role="assistant"),
                    index=choice.index,
                )
            ]
        )

    def _try_run_function(self, choice: Choice) -> llm.ChatChunk | None:
        if not self._fnc_ctx:
            logger.warning("oai stream tried to run function without function context")
            return None

        if self._tool_call_id is None:
            logger.warning(
                "oai stream tried to run function but tool_call_id is not set"
            )
            return None

        if self._fnc_name is None or self._fnc_raw_arguments is None:
            logger.warning(
                "oai stream tried to call a function but raw_arguments and fnc_name are not set"
            )
            return None

        fnc_info = llm._oai_api.create_ai_function_info(
            self._fnc_ctx, self._tool_call_id, self._fnc_name, self._fnc_raw_arguments
        )
        self._tool_call_id = self._fnc_name = self._fnc_raw_arguments = None
        self._function_calls_info.append(fnc_info)

        return llm.ChatChunk(
            choices=[
                llm.Choice(
                    delta=llm.ChoiceDelta(role="assistant", tool_calls=[fnc_info]),
                    index=choice.index,
                )
            ]
        )


def _build_oai_context(
    chat_ctx: llm.ChatContext, cache_key: Any
) -> list[ChatCompletionMessageParam]:
    return [_build_oai_message(msg, cache_key) for msg in chat_ctx.messages]  # type: ignore


def _build_oai_message(msg: llm.ChatMessage, cache_key: Any):
    oai_msg: dict = {"role": msg.role}

    if msg.name:
        oai_msg["name"] = msg.name

    # add content if provided
    if isinstance(msg.content, str):
        oai_msg["content"] = msg.content
    elif isinstance(msg.content, list):
        oai_content = []
        for cnt in msg.content:
            if isinstance(cnt, str):
                oai_content.append({"type": "text", "text": cnt})
            elif isinstance(cnt, llm.ChatImage):
                oai_content.append(_build_oai_image_content(cnt, cache_key))

        oai_msg["content"] = oai_content

    # make sure to provide when function has been called inside the context
    # (+ raw_arguments)
    if msg.tool_calls is not None:
        tool_calls: list[dict[str, Any]] = []
        oai_msg["tool_calls"] = tool_calls
        for fnc in msg.tool_calls:
            tool_calls.append(
                {
                    "id": fnc.tool_call_id,
                    "type": "function",
                    "function": {
                        "name": fnc.function_info.name,
                        "arguments": fnc.raw_arguments,
                    },
                }
            )

    # tool_call_id is set when the message is a response/result to a function call
    # (content is a string in this case)
    if msg.tool_call_id:
        oai_msg["tool_call_id"] = msg.tool_call_id

    return oai_msg


def _build_oai_image_content(image: llm.ChatImage, cache_key: Any):
    if isinstance(image.image, str):  # image url
        return {
            "type": "image_url",
            "image_url": {"url": image.image, "detail": "auto"},
        }
    elif isinstance(image.image, rtc.VideoFrame):  # VideoFrame
        if cache_key not in image._cache:
            # inside our internal implementation, we allow to put extra metadata to
            # each ChatImage (avoid to reencode each time we do a chatcompletion request)
            opts = utils.images.EncodeOptions()
            if image.inference_width and image.inference_height:
                opts.resize_options = utils.images.ResizeOptions(
                    width=image.inference_width,
                    height=image.inference_height,
                    strategy="center_aspect_fit",
                )

            encoded_data = utils.images.encode(image.image, opts)
            image._cache[cache_key] = base64.b64encode(encoded_data).decode("utf-8")
            print('======image opts======', opts)

        return {
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{image._cache[cache_key]}"},
        }

    raise ValueError(f"unknown image type {type(image.image)}")
