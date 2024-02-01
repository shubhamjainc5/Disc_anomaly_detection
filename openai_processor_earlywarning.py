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
import time
import copy
import logging
Logger = logging.getLogger('openai-earlywarning')
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

from prompt import  forecasting_sys_prompt, forecasting_prompt


with open("config.json", "r") as f:
    domain_config = json.load(f)["openai_cred"]


# Set llm_cache
globals.set_llm_cache(SQLiteCache(database_path="ew_cache.db"))


os.environ["OPENAI_API_TYPE"] = domain_config['OPENAI_API_TYPE']
os.environ["OPENAI_API_VERSION"] = domain_config['OPENAI_API_VERSION']
os.environ["OPENAI_API_BASE"] = domain_config['OPENAI_API_BASE']
os.environ["OPENAI_API_KEY"] = domain_config['OPEN_AI_API_KEY']


class Narrative(BaseModel):

    anomaly_date: str = Field(description="anomaly dates which were present in table data")
    narrative: str = Field(description="narrative with respect to selected anomaly date")
    narrative_html: str = Field(description="narrative in html format")


class ForecastNarrativePlainText(BaseModel):

    ForecastNarrative: List[Narrative] = Field(description="a set of anomalous points with their date and it's narrative in dictionary format")


def generate_ew_narrative(table_data:str,anomaly_dates:List, use_cache:bool) -> (List[Dict[str,str]], int, float):
    print('\n---------Early warning-------------\n')

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

        system_message_prompt = SystemMessagePromptTemplate.from_template(forecasting_sys_prompt)

        reason_parser = PydanticOutputParser(pydantic_object=ForecastNarrativePlainText)
        reason_format_instructions = reason_parser.get_format_instructions()

        reason_template = PromptTemplate(input_variables=["table_data"], template=forecasting_prompt,
                                        partial_variables={"format_instructions": reason_format_instructions})
        
        reason_prompt = reason_template.format(table_data=table_data, format_instructions = reason_format_instructions)
        # print(reason_prompt)

        human_message_prompt = HumanMessagePromptTemplate(prompt=reason_template)

        chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])

        reason_chain = LLMChain(llm=model, prompt=chat_prompt, output_key="generated_narrative")

        start_time = time.time()
        gpt_op = reason_chain({"table_data": table_data})
        Logger.info("\n Response from GPT : {0}".format(gpt_op))
        end_time = time.time()
        Logger.info("LLM output took {:.4f} seconds".format(end_time - start_time))

        generated_reasons = gpt_op['generated_narrative']
        # print(generated_reasons)

        start_time = time.time()
        reason_ip_tokens = model.get_num_tokens(reason_prompt)
        reason_op_tokens = model.get_num_tokens(generated_reasons)
        end_time = time.time()
        Logger.info("token calculation took {:.4f} seconds".format(end_time - start_time))
        call_tokens.append(reason_ip_tokens + reason_op_tokens)
        cnt += 1

        try:
            reason_result = reason_parser.parse(generated_reasons).json()
            reason_result = json.loads(reason_result)['ForecastNarrative']
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
