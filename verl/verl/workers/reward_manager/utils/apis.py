import logging
import json
import re
import random
import requests
import time
import math
import uuid
import pandas as pd
import backoff
import concurrent.futures
import openai
import os
import sys


SELF_VERIFIER_SERVER = os.environ.get('SELF_VERIFIER_SERVER')
SELF_VERIFIER_SERVER_NAME = os.environ.get('SELF_VERIFIER_SERVER_NAME')

if not SELF_VERIFIER_SERVER or not SELF_VERIFIER_SERVER_NAME:
    print("error: SELF_VERIFIER_SERVER not found")
    exit(-1)


def request_verifier(messages):
    client = openai.Client(base_url=f"http://{SELF_VERIFIER_SERVER}/v1", api_key="EMPTY")
    for _ in range(5):
        try:
            response = client.chat.completions.create(
                model=SELF_VERIFIER_SERVER_NAME,
                messages=messages,
                temperature=0.1,
                max_tokens=8192
            )
            return response.choices[0].message.content

        except:
            time.sleep(1)
            continue
            
    return ""