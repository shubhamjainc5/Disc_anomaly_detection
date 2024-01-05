"""
This script prepares request for GPT-3 APIs, Hits the API and returns the response.
@author: Sujith R Kumar

Returns:
    _type_: _description_
"""

import json
import numpy as np
import os
import re
import copy
from logging_handler import Logger
from langchain.schema import OutputParserException
# openai is commented to avoid langchain resource not found error
#import openai

from typing import List, Dict
# from utils.timer import timed
import traceback

from json.decoder import JSONDecodeError
from openai.error import APIError, Timeout, RateLimitError, APIConnectionError, AuthenticationError, \
    ServiceUnavailableError, InvalidRequestError

from langchain.llms import AzureOpenAI
from langchain.chat_models import AzureChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
# from langchain.chains import SequentialChain, TransformChain
# from langchain.memory import SimpleMemory
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.cache import SQLiteCache
from langchain import globals

from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field, validator
from typing import List

import ast

from prompt import  reasoning_sys_prompt, reasoning_prompt


with open("config.json", "r") as f:
    domain_config = json.load(f)["openai_cred"]


# Set llm_cache
globals.set_llm_cache(SQLiteCache(database_path="cache.db"))


os.environ["OPENAI_API_TYPE"] = domain_config['OPENAI_API_TYPE']
os.environ["OPENAI_API_VERSION"] = domain_config['OPENAI_API_VERSION']
os.environ["OPENAI_API_BASE"] = domain_config['OPENAI_API_BASE']
os.environ["OPENAI_API_KEY"] = domain_config['OPEN_AI_API_KEY']


class Reasoning(BaseModel):

    anomaly_date: str = Field(description="anomaly dates which were present in table data")
    reason: str = Field(description="casual reasoning with respect to selected anomaly date")
    reason_html: str = Field(description="casual reasoning in html format")


class AnomalyReasoningPlainText(BaseModel):

    AnomalyReasoning: List[Reasoning] = Field(description="a set of anomalous points with their date and it's casual reasoning in dictionary format")


def generate_reasoning(table_data:str, anomaly_dates:List, use_cache:bool) -> (List[Dict[str,str]], int, float):
    print('\n---------Casual Reasoning-------------\n')

    cnt = 0
    call_tokens = []

    def reason_llm_call() -> List[Dict[str,str]]:

        nonlocal cnt

        model = AzureChatOpenAI(
            deployment_name=domain_config['AZURE_DEPLOYMENT_NAME'],
            model_name=domain_config['AZURE_MODEL_NAME'],
            max_tokens=2000,
            temperature=0.5,
            verbose=True,
            request_timeout=20,
            cache= use_cache
        )

        system_message_prompt = SystemMessagePromptTemplate.from_template(reasoning_sys_prompt)

        reason_parser = PydanticOutputParser(pydantic_object=AnomalyReasoningPlainText)
        reason_format_instructions = reason_parser.get_format_instructions()

        reason_template = PromptTemplate(input_variables=["table_data","anomaly_dates"], template=reasoning_prompt,
                                        partial_variables={"format_instructions": reason_format_instructions})
        
        reason_prompt = reason_template.format(table_data=table_data, anomaly_dates= str(anomaly_dates), format_instructions = reason_format_instructions)
        # print(reason_prompt)

        human_message_prompt = HumanMessagePromptTemplate(prompt=reason_template)

        chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])

        reason_chain = LLMChain(llm=model, prompt=chat_prompt, output_key="generated_reasoning")

        gpt_op = reason_chain({"table_data": table_data, "anomaly_dates": str(anomaly_dates)})
        Logger.info("\n Response from GPT : {0}".format(gpt_op))

        generated_reasons = gpt_op['generated_reasoning']
        # print(generated_reasons)

        reason_ip_tokens = model.get_num_tokens(reason_prompt)
        reason_op_tokens = model.get_num_tokens(generated_reasons)
        call_tokens.append(reason_ip_tokens + reason_op_tokens)
        cnt += 1

        try:
            reason_result = reason_parser.parse(generated_reasons).json()
            reason_result = json.loads(reason_result)['AnomalyReasoning']
            Logger.info('\n Langchain json reason content parsed: {0}'.format(reason_result))
        except:
            reason_result = generated_reasons[generated_reasons.find('['): generated_reasons.rfind(']') + 1]
            reason_result = ast.literal_eval(reason_result)
            Logger.info('\n Regex json reason content parsed: {0}'.format(reason_result))

        gpt_dt = [ res["anomaly_date"] for res in reason_result]

        if set(gpt_dt) == set(anomaly_dates) and len(gpt_dt) == len(anomaly_dates):
            pass
        else:
            raise AssertionError(f"Count of anamolous dates in LLM output does not match with provided dates.")

        return reason_result

    try:
        result = reason_llm_call()
        status = 'LLMParsed'
    except (AssertionError, JSONDecodeError, OutputParserException, KeyError) as e:
        Logger.error(traceback.format_exc())
        Logger.error('ERROR: reason Parsing or Assertion Failed')
        try:
            globals.get_llm_cache().clear()
            result = reason_llm_call()
            status = 'LLMRetryParsed'
        except (APIError, Timeout, ServiceUnavailableError) as e1:
            Logger.error(traceback.format_exc())
            Logger.error('ERROR: LLM SERVER Not responding')
            result = []
            status = 'LLMServerError'
            globals.get_llm_cache().clear()
        except (RateLimitError, APIConnectionError, InvalidRequestError, AuthenticationError) as e3:
            Logger.error(traceback.format_exc())
            Logger.error('ERROR: Application server Not responding')
            result = []
            status = 'AppServerError'
            globals.get_llm_cache().clear()
        except:
            Logger.error(traceback.format_exc())
            Logger.error('UNKNOWN ERROR: occurred')
            result = []
            status = 'UnknownError'
            globals.get_llm_cache().clear()
    except (APIError, Timeout, ServiceUnavailableError) as e1:
        Logger.error(traceback.format_exc())
        Logger.error('ERROR: LLM SERVER Not responding')
        result = []
        status = 'LLMServerError'
        globals.get_llm_cache().clear()
    except (RateLimitError, APIConnectionError, InvalidRequestError, AuthenticationError) as e3:
        Logger.error(traceback.format_exc())
        Logger.error('ERROR: Application server Not responding')
        result = []
        status = 'AppServerError'
        globals.get_llm_cache().clear()
    except:
        Logger.error(traceback.format_exc())
        Logger.error('UNKNOWN ERROR: occurred')
        result = []
        status = 'UnknownError'
        globals.get_llm_cache().clear()

    print('\n---------reason:{0}-------------\n'.format(status))
    Logger.info(f"\n---------reason:{status}-------------\n")

    return result, cnt, 0 if len(call_tokens) == 0 else np.average(call_tokens), status
